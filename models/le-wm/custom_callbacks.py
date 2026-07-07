
from lightning.pytorch.callbacks import Callback
import torch
import torch.nn as nn
from captum.attr import IntegratedGradients

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr
import pandas as pd
import os
import wandb


class PredictorFeatureImportanceWrapper(torch.nn.Module):
    def __init__(self, predictor):
        super().__init__()
        self.predictor = predictor
        
    def forward(self, ctx_emb, ctx_act):
        """
        ctx_emb:  [B, ctx_len, D] -> ViT patch context embeddings
        ctx_act:  [B, ctx_len, Act_D] -> Action context embeddings
        """
        # 1. Run the predictor forward pass
        pred_emb = self.predictor(ctx_emb, ctx_act) # Shape: [B, n_preds, D]
        
        # 2. Convert the entire output tensor into a scalar energy value.
        # We use a Frobenius-like norm (sum of squares).
        # A change in an input feature that vastly changes this sum has high importance.
        scalar_energy = torch.sum(pred_emb ** 2).view(1)
        
        return scalar_energy
    


class LeWMPredictorIGCallback(Callback):
    def __init__(self, every_n_epochs: int = 1, n_steps: int = 30, ctx_len: int = 1):
        super().__init__()
        self.every_n_epochs = every_n_epochs
        self.n_steps = n_steps
        self.ctx_len = ctx_len
        self.ig = None
        self.wrapper = None

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if batch_idx != 0 or (trainer.current_epoch + 1) % self.every_n_epochs != 0:
            return

        if self.ig is None:
            self.wrapper = PredictorFeatureImportanceWrapper(pl_module.model.predict)
            self.ig = IntegratedGradients(self.wrapper)

        # Clean NaNs just like the main forward pass
        batch["action"] = torch.nan_to_num(batch["action"], 0.0)

        with torch.enable_grad():
            # Run the visual and action encoders
            output = pl_module.model.encode(batch)
            emb = output["emb"]
            act_emb = output["act_emb"]

            # Isolate and clone the exact context slices
            ctx_emb = emb[:, :self.ctx_len].detach().clone().requires_grad_(True)
            ctx_act = act_emb[:, :self.ctx_len].detach().clone().requires_grad_(True)

            # Baselines
            ctx_emb_baseline = torch.zeros_like(ctx_emb)
            ctx_act_baseline = torch.zeros_like(ctx_act)

            # Compute Attributions
            attributions = self.ig.attribute(
                inputs=(ctx_emb, ctx_act),
                baselines=(ctx_emb_baseline, ctx_act_baseline),
                n_steps=self.n_steps
            )
            
            emb_attr, act_attr = attributions

        # ---- Aggregate Metrics over the context window ----
        # Calculate overall visual feature vs action feature importance
        vit_importance = torch.abs(emb_attr).sum().item()
        action_importance = torch.abs(act_attr).sum().item()
        total = vit_importance + action_importance + 1e-8
        
        vit_pct = (vit_importance / total) * 100
        action_pct = (action_importance / total) * 100

        logger = trainer.logger
        if logger:
            # Bypasses lightning's batch averaging and logs raw step values
            logger.log_metrics(
                metrics={
                    "predictor/vit_importance_pct": vit_pct,
                    "predictor/action_importance_pct": action_pct,
                },
                step=trainer.global_step
            )




class LinearProbeCallback(Callback):
    """
    Train a linear probe on frozen encoder latents.
        every_n_epochs: how often the probes should be fit
        latent_key: later can also be switched to "cls" rather than "emb"

    Probe will run "every_n_epochs" on_validation_epoch_end but also once 
    at the end of the full training

    """

    def __init__(self, every_n_epochs=25, max_train_batches=15, max_val_batches=10, target_slice=slice(22, 26), latent_key="emb"):
        super().__init__()

        self.every_n_epochs = every_n_epochs
        self.max_train_batches = max_train_batches
        self.max_val_batches = max_val_batches
        self.target_slice = target_slice
        self.latent_key=latent_key

        self.probes = {
            "joint_position": lambda b: b["observation"][..., 0:6],
            "joint_velocity": lambda b: b["observation"][..., 6:12],
            "ee_position": lambda b: b["observation"][..., 12:15],
            "ee_yaw": lambda b: b["observation"][..., 15:17],  
            "gripper_opening": lambda b: b["observation"][...,17:18],
            "gripper_contact": lambda b: b["observation"][...,18:19],
            "block_position": lambda b: b["observation"][..., 19:22],
            "block_quaternion": lambda b: b["observation"][..., 22:26],
            "block_yaw": lambda b: b["observation"][..., 26:28],
        }

    @torch.no_grad()
    def collect_latents(self, loader, pl_module, max_batches):
        X = []
        Y = {name: [] for name in self.probes}

        device = pl_module.device
        pl_module.model.eval()

        for batch_idx, batch in enumerate(loader):
            if batch_idx >= max_batches:
                break

            batch = {
                k: v.to(device) if torch.is_tensor(v) else v
                for k, v in batch.items()
            }

            batch["action"] = torch.nan_to_num(batch["action"], 0.0)
            output = pl_module.model.encode(batch)

            emb = output[self.latent_key]                     # (B,T,D)

            X.append(emb.reshape(-1, emb.shape[-1]).cpu().numpy())
            for name, target_fn in self.probes.items():
                target = target_fn(batch)
                Y[name].append(target.reshape(-1, target.shape[-1]).cpu().numpy())

        X = np.concatenate(X, axis=0)
        for name in Y:
            Y[name] = np.concatenate(Y[name], axis=0)

        return X, Y
    
    def run_probe_model(self, X_train, Y_train, X_val, Y_val, probe_constructor, prefix): 
        metrics = {}
        rows = []
        for name in self.probes:
            probe_model = probe_constructor()

            target_train = Y_train[name]

            if target_train.shape[1] == 1:
                target_train = target_train.ravel()

            probe_model.fit(X_train, target_train)
            pred = probe_model.predict(X_val)

            pred = np.atleast_2d(pred).T if pred.ndim == 1 else pred
            target = np.atleast_2d(Y_val[name]).T if Y_val[name].ndim == 1 else Y_val[name]

            mse = mean_squared_error(target,pred)
            r2 = r2_score(target,pred)

            pearsons = []
            for i in range(target.shape[1]):
                r, _ = pearsonr(target[:, i], pred[:, i])
                pearsons.append(r)

            metrics[f"{prefix}_probe/{name}/mse"] = mse
            metrics[f"{prefix}_probe/{name}/r2"] = r2
            metrics[f"{prefix}_probe/{name}/pearson"] = np.mean(pearsons)

            if name == "block_position": 
                print("Probe results for block position")
                print(f"Probe MSE      : {mse:.6f}")
                print(f"Probe R2       : {r2:.4f}")
                print(f"Probe Pearson r: {np.mean(pearsons):.4f}")

            rows.append({
                "probe": name,
                "model": prefix,
                "mse": mse,
                "r2": r2,
                "pearson": np.mean(pearsons),
            })

        return metrics, rows

    def on_validation_epoch_end(self, trainer, pl_module):

        epoch = trainer.current_epoch
        if (epoch == 0) or (epoch % self.every_n_epochs != 0):
            return

        print(f"\nRUNNING LINEAR PROBE (epoch {epoch})")
        train_loader = trainer.train_dataloader
        val_loader = trainer.val_dataloaders

        X_train, Y_train = self.collect_latents(train_loader, pl_module, self.max_train_batches)
        X_val, Y_val = self.collect_latents(val_loader, pl_module, self.max_val_batches)

        metrics, _ = self.run_probe_model(X_train=X_train, Y_train=Y_train, X_val=X_val, Y_val=Y_val, probe_constructor=LinearRegression, prefix="LINEAR")

        trainer.logger.log_metrics(
            metrics,
            step=trainer.global_step,
        )

    def on_train_end(self, trainer, pl_module):
        train_loader = trainer.train_dataloader
        val_loader = trainer.val_dataloaders

        X_train, Y_train = self.collect_latents(train_loader, pl_module, self.max_train_batches)
        X_val, Y_val = self.collect_latents(val_loader, pl_module, self.max_val_batches)

        linear_metrics, linear_rows = self.run_probe_model(X_train=X_train, Y_train=Y_train, X_val=X_val, Y_val=Y_val, probe_constructor=LinearRegression, prefix="LINEAR")
        trainer.logger.log_metrics(linear_metrics,step=trainer.global_step)

        MLP_probe = lambda: MLPRegressor(
            hidden_layer_sizes=(256, 128),
            activation="relu",
            max_iter=1000,
            random_state=42,
        )
        MLP_metrics, MLP_rows = self.run_probe_model(X_train=X_train, Y_train=Y_train, X_val=X_val, Y_val=Y_val, probe_constructor=MLP_probe, prefix="MLP")
        trainer.logger.log_metrics(MLP_metrics,step=trainer.global_step)
        
        rows = linear_rows + MLP_rows
        df = pd.DataFrame(rows)
        trainer.logger.log_table(key="probe_results",dataframe=df,)

        combined_df = (
            df.set_index(["probe", "model"])
            .unstack("model")
            .swaplevel(0, 1, axis=1)
        )
        combined_df = combined_df.copy()

        combined_df.columns = [
            f"{model}_{metric}"
            for model, metric in combined_df.columns
        ]

        combined_df = combined_df.reset_index()
        trainer.logger.log_table(key="probe_results_combined",dataframe=combined_df,)

        # # Save tidy CSV
        # run_dir = trainer.logger.experiment.dir
        # csv_path = os.path.join(run_dir, "probe_results.csv")
        # df.to_csv(csv_path, index=False)

        

        # # Log interactive table to WandB
        # run_dir.log({"probe_results": wandb.Table(dataframe=df)})

        # Create a pretty table
        # table = (
        #     df.set_index(["probe", "model"])
        #     .unstack("model")
        #     .swaplevel(0, 1, axis=1)
        # )
        # table.to_markdown(os.path.join(trainer.logger.log_dir, "probe_results.md"))

    
import torch
import pytorch_lightning as pl
class RandomActionLatentMSECallback(Callback):
    """
    Compare predictor outputs using the true actions vs shuffled actions.
    """

    def __init__(self, every_n_epochs=1, ctx_len=1):
        super().__init__()
        self.every_n_epochs = every_n_epochs
        self.ctx_len = ctx_len

    @torch.no_grad()
    def on_validation_batch_end(
        self,
        trainer,
        pl_module,
        outputs,
        batch,
        batch_idx,
    ):
        if batch_idx != 0:
            return

        if trainer.current_epoch % self.every_n_epochs != 0:
            return

        device = pl_module.device

        batch = {
            k: v.to(device) if torch.is_tensor(v) else v
            for k, v in batch.items()
        }

        batch["action"] = torch.nan_to_num(batch["action"], 0.0)

        # ------------------------------
        # Encode once
        # ------------------------------
        output = pl_module.model.encode(batch)

        emb = output["emb"]
        act_emb = output["act_emb"]

        ctx_emb = emb[:, :self.ctx_len]
        ctx_act = act_emb[:, :self.ctx_len]

        # ------------------------------
        # Prediction with true actions
        # ------------------------------
        pred_true = pl_module.model.predict(
            ctx_emb,
            ctx_act,
        )

        # ------------------------------
        # Prediction with shuffled actions
        # ------------------------------
        perm = torch.randperm(ctx_act.shape[0], device=device)

        pred_random = pl_module.model.predict(
            ctx_emb,
            ctx_act[perm],
        )

        mse = torch.mean((pred_true - pred_random) ** 2)

        trainer.logger.log_metrics(
            {
                "predictor/random_action_latent_mse": mse.item(),
            },
            step=trainer.global_step,
        )