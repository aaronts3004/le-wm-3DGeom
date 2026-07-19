
from lightning.pytorch.callbacks import Callback
import torch
import torch.nn as nn
from captum.attr import IntegratedGradients

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr
from stable_worldmodel.wm.utils import save_pretrained
import pandas as pd
import os
import wandb
import time
from einops import repeat
import matplotlib.pyplot as plt
import umap
from sklearn.manifold import TSNE

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
    

class LatentTSENPlots(Callback):
    def __init__(self, latent_plot_dir): 
        super().__init__() 
        self.output_dir = latent_plot_dir 

    @torch.no_grad()
    def compute_latent_plots(self, val_loader, pl_module):

        device = pl_module.device
        pl_module.model.eval()

        pred_latents = []
        gt_states = []

        MAX_BATCHES = 100

        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                enc = pl_module.model.encode(batch) # [B,T,D]     
                pred_latents.append(enc["emb"].reshape(-1, 192).cpu().numpy())

                obs = batch["observation"]      # [B,26]
                print("obs.shape=", obs.shape)
                physical_state = torch.cat([
                    obs[:, :, 12:15],          # ee    
                    obs[:, :, 19:22],          # cube
                ], dim=-1)

                gt_states.append(physical_state.reshape(-1, 6).cpu().numpy())
                print("encoded batch=", i)

                if i >= MAX_BATCHES:
                    break

        print("Done with latent encoding")
        
        pred_latents = np.concatenate(pred_latents)   # [N, D]
        gt_states    = np.concatenate(gt_states)      # [N, 6]

        N = pred_latents.shape[0]
        print(f"Computed N={N} latents")
        print("pred_latents.shape=", pred_latents.shape)
        print("gt_state.shape=", gt_states.shape)

        idx1 = torch.randint(0,N,(10000,))
        idx2 = torch.randint(0,N,(10000,))

        d_lat = np.linalg.norm(
            pred_latents[idx1] - pred_latents[idx2],
            axis=1,
        )

        d_state = np.linalg.norm(
            gt_states[idx1] - gt_states[idx2],
            axis=1,
        )
        r, p = pearsonr(d_lat,d_state)
        print("PEARSON=", r)

        plt.figure(figsize=(6,6))

        plt.scatter(
            d_state,
            d_lat,
            s=2,
            alpha=0.15,
        )

        plt.xlabel("Physical distance")
        plt.ylabel("Latent distance")
        plt.title(f"Pearson r = {r:.3f}")
        plt.savefig(f"{self.output_dir}/distance_scatter.png")
        plt.close() 

        plt.figure()

        plt.hist(
            d_lat,
            bins=100,
        )
        plt.savefig(f"{self.output_dir}/latent_distance_hist.png")
        plt.close()

        embedding = umap.UMAP().fit_transform(pred_latents)
        plt.scatter(
            embedding[:,0],
            embedding[:,1],
            c=gt_states[:,3],   # cube x
            s=3,
            cmap="viridis",
        )
        plt.savefig(f"{self.output_dir}/UMAP.png")
        plt.close()

        for perplexity in [1,10,20,30,50]:

            tsne = TSNE(
                n_components=2,
                perplexity=perplexity,
                learning_rate="auto",
                init="pca",
                random_state=42,
            )

            latent_2d = tsne.fit_transform(
                pred_latents
            )

            titles = [
                "EE X",
                "EE Y",
                "Block X",
                "Block Y",
            ]

            for i, t in enumerate(titles):

                plt.figure(figsize=(7,6))
                plt.scatter(
                    latent_2d[:,0],
                    latent_2d[:,1],
                    c=gt_states[:,i],   # state (i)
                    s=5,
                    alpha=0.7,
                    cmap="viridis",
                )

                plt.colorbar(label=f"t")
                plt.title(f"t-SNE colored by {t}")
                plt.xlabel("t-SNE 1")
                plt.ylabel("t-SNE 2")
                plt.savefig(f"{self.output_dir}/cube_tsne_{t}_perpl_{perplexity}.png")

    

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
        latent_key: by default, we train on the "emb" latents from JEPA.encode(), 
            which corresponds to the projected CLS token. Adjusting JEPA.encode(), 
            we can also decide to train the probes on a different latent embedding
        model: on which model we are doing the probing. Options: "jepa" or "prejepa"
        vit_pooling: how to downsample the patch tokens of prejepa vit encoder. Options: "mean" or "spatial_2x2"

    Probe will run "every_n_epochs" on_validation_epoch_end but also once 
    at the end of the full training

    """

    def __init__(self, every_n_epochs=25, max_train_batches=15, max_val_batches=10, latent_key="emb", model="jepa", vit_pooling="mean"):
        super().__init__()

        self.every_n_epochs = every_n_epochs
        self.max_train_batches = max_train_batches
        self.max_val_batches = max_val_batches
        self.latent_key = latent_key
        self.model_struc = model
        self.pooling_mode = vit_pooling
        self.wm_history_size = None                     # TODO

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

            if self.model_struc == "jepa":
                emb = output[self.latent_key]                     # (B,T,D)
                X.append(emb.reshape(-1, emb.shape[-1]).cpu().numpy())
                
            elif self.model_struc == "prejepa":
                emb = output["pixels_emb"] 
                B, T, P, D = emb.shape

                if self.pooling_mode == "mean":
                    emb_global = emb.mean(dim=2)  # (B, T, 384)
                    X_batch = emb_global.reshape(B * T, -1)

                elif self.pooling_mode == "spatial_2x2":
                    emb_spatial = emb.view(B * T, 16, 16, D).permute(0, 3, 1, 2) # (B*T, D, 16, 16)
                    emb_downsampled = nn.functional.adaptive_avg_pool2d(emb_spatial, (2, 2)) # (B*T, D, 2, 2)
                    X_batch = emb_downsampled.permute(0, 2, 3, 1).reshape(B * T, -1) 

                X.append(X_batch.cpu().numpy())

            for name, target_fn in self.probes.items():
                target = target_fn(batch)
                Y[name].append(target.reshape(-1, target.shape[-1]).cpu().numpy())

        X = np.concatenate(X, axis=0)
        for name in Y:
            Y[name] = np.concatenate(Y[name], axis=0)

        return X, Y

    def compute_probe_metrics(self, target, pred):
        pred = np.atleast_2d(pred).T if pred.ndim == 1 else pred
        target = np.atleast_2d(target).T if target.ndim == 1 else target

        mse = mean_squared_error(target, pred)
        r2 = r2_score(target, pred)

        pearsons = [
            pearsonr(target[:, i], pred[:, i])[0]
            for i in range(target.shape[1])
        ]

        return {
            "mse": mse,
            "r2": r2,
            "pearson": np.mean(pearsons),
        }
    
    
    def run_probe_model(self, X_train, Y_train, X_val, Y_val, probe_constructor, prefix, seed): 
        metrics = {}
        rows = []
        for name in self.probes:
            probe_model = probe_constructor()

            target_train = Y_train[name]

            if target_train.shape[1] == 1:
                target_train = target_train.ravel()

            probe_model.fit(X_train, target_train)

            train_metrics = self.compute_probe_metrics(target_train, probe_model.predict(X_train))
            val_metrics = self.compute_probe_metrics(Y_val[name], probe_model.predict(X_val))

            for split, split_metrics in [
                ("train", train_metrics),
                ("val", val_metrics),
            ]:
                for metric_name, value in split_metrics.items():
                    metrics[f"{prefix}_probe/{split}/{name}/{metric_name}"] = value

            rows.append({
                "probe": name,
                "model": prefix,
                "split": "train",
                "seed": seed,
                **train_metrics,        # dict with mse, r2 and pearson
            })

            rows.append({
                "probe": name,
                "model": prefix,
                "split": "val",
                "seed": seed,
                **val_metrics,          # dict with mse, r2 and pearson
            })

        return metrics, rows

    def run_probe_call(self, train_loader, val_loader, trainer, pl_module):
        ''' 
            Enables the probing to be started from an arbitrary point in the code, (e.g. after loading a CKPT)
            without relying on Lightning callback moments (e.g. on_validation_end)
        '''
        logger = trainer.logger
        print("\n\nRUNNING PROBES OUTSIDE TRAINING (with dedicated call)")

        X_train, Y_train = self.collect_latents(train_loader, pl_module, self.max_train_batches)
        X_val, Y_val = self.collect_latents(val_loader, pl_module, self.max_val_batches)

        # ==============================
        # for random baseline comparison
        # comment out to get real latent probing 
        # ==============================
        # for name in Y_val.keys(): 
        #     perm = np.random.permutation(len(Y_train[name]))
        #     Y_train[name] = Y_train[name][perm]
        # ==============================

        print(f"Training on X_train.shape={X_train.shape}")
        print(f"Validating on X_val.shape={X_val.shape}")

        metrics, rows = self.run_probe_model(X_train=X_train, Y_train=Y_train, X_val=X_val, Y_val=Y_val, probe_constructor=LinearRegression, prefix="START_LINEAR")

        trainer.logger.log_metrics(metrics,step=trainer.global_step,)
        df = pd.DataFrame(rows)
        trainer.logger.log_table(key="probe_start",dataframe=df,)


        print("\n\nFINISHED PROBE FIT")

    def on_validation_epoch_end(self, trainer, pl_module):

        device = pl_module.device

        ### 1. Get a large train-val split from the dataloaders and fit the probe on this data

        epoch = trainer.current_epoch
        if (epoch % self.every_n_epochs != 0):
            return
        
        t0 = time.time()
        print(f"\n\nRUNNING LINEAR PROBE (epoch {epoch})")

        train_loader = trainer.train_dataloader
        val_loader = trainer.val_dataloaders

        X_train, Y_train = self.collect_latents(train_loader, pl_module, self.max_train_batches)
        X_val, Y_val = self.collect_latents(val_loader, pl_module, self.max_val_batches)

        print(f"Training on X_train.shape={X_train.shape}")
        print(f"Validating on X_val.shape={X_val.shape}")

        metrics, _ = self.run_probe_model(X_train=X_train, Y_train=Y_train, X_val=X_val, Y_val=Y_val, probe_constructor=LinearRegression, prefix="LINEAR", seed=0)
        trainer.logger.log_metrics(metrics, step=trainer.global_step)
        
        t1 = time.time() 
        print(f"Probe ran in: {(t1-t0)} seconds")

        ### 2. Freeze the linear probe and perform a latent rollout (on a single / longer sample)
        # full_dataset = trainer.train_dataloader.dataset
        # episode = full_dataset.load_episode(0)          # always load episode 0 - returns [T, H, W, C] where T is total number of frame-skipped time-steps

        
    def on_train_end(self, trainer, pl_module):
        logger = trainer.logger
        t0 = time.time()
        print(f"\n\nRUNNING LINEAR PROBE at the end of training, final run")

        train_loader = trainer.train_dataloader
        val_loader = trainer.val_dataloaders

        X_train, Y_train = self.collect_latents(train_loader, pl_module, self.max_train_batches)
        X_val, Y_val = self.collect_latents(val_loader, pl_module, self.max_val_batches)

        ### Collect latents once, then probe 5 times
        N_RUNS=5

        all_rows = []
        all_metrics = []

        for seed in range(N_RUNS):
            LinearProbe = lambda: LinearRegression()
            metrics, rows = self.run_probe_model(
                X_train=X_train,
                Y_train=Y_train,
                X_val=X_val,
                Y_val=Y_val,
                probe_constructor=LinearProbe,
                prefix="LINEAR",
                seed=seed,          # pass into your probe
            )
            all_rows.extend(rows)
            all_metrics.append(metrics)

            if seed == 0:       # log the first probe only in the line plot
                trainer.logger.log_metrics(metrics,step=trainer.global_step)


        # linear_metrics, linear_rows = self.run_probe_model(X_train=X_train, Y_train=Y_train, X_val=X_val, Y_val=Y_val, probe_constructor=LinearRegression, prefix="LINEAR")
        # trainer.logger.log_metrics(linear_metrics,step=trainer.global_step)

        for seed in range(N_RUNS):
            MLP_probe = lambda s=seed: MLPRegressor(
                hidden_layer_sizes=(256,128),
                activation="relu",
                max_iter=1000,
                random_state=s,
            )

            metrics, rows = self.run_probe_model(X_train=X_train, Y_train=Y_train, X_val=X_val, Y_val=Y_val, probe_constructor=MLP_probe, prefix="MLP", seed=seed)
            all_rows.extend(rows)

            if seed == 0: 
                trainer.logger.log_metrics(metrics,step=trainer.global_step)



        summary_df = (
            pd.DataFrame(all_rows)
            .groupby(["probe", "split", "model"])
            .agg(
                mse_mean=("mse", "mean"),
                mse_std=("mse", "std"),
                r2_mean=("r2", "mean"),
                r2_std=("r2", "std"),
                pearson_mean=("pearson", "mean"),
                pearson_std=("pearson", "std"),
            )
            .reset_index()
        )
        summary_df["R2"] = summary_df.apply(
            lambda r: f"{r.r2_mean:.3f} ± {r.r2_std:.3f}", axis=1
        )

        summary_df["MSE"] = summary_df.apply(
            lambda r: f"{r.mse_mean:.3f} ± {r.mse_std:.3f}", axis=1
        )

        summary_df["Pearson"] = summary_df.apply(
            lambda r: f"{r.pearson_mean:.3f} ± {r.pearson_std:.3f}", axis=1
        )

        raw_df = pd.DataFrame(all_rows)
        trainer.logger.log_table(
            key="probe_results_raw",
            dataframe=raw_df,          # one row per seed
        )

        trainer.logger.log_table(
            key="probe_results_summary",
            dataframe=summary_df,  # mean/std
        )
   
        t1 = time.time() 
        print(f"Finished Final Probing in {(t1-t0)} seconds")

    
import torch
import pytorch_lightning as pl
import torch.nn.functional as F
class RandomActionLatentMSECallback(Callback):
    """
    Compare predictor outputs using the true actions vs shuffled actions.
    """

    def __init__(self, every_n_epochs=1, ctx_len=1, model="jepa"):
        super().__init__()
        self.every_n_epochs = every_n_epochs
        self.ctx_len = ctx_len
        self.model_struc = model

    def predict_jepa_emd(self, pl_module, batch, device):
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

        emb_true = emb

        return emb_true, pred_true, pred_random
    
    def predict_prejepa_emd(self, pl_module, batch, device):
        # ------------------------------
        # Encode once
        # ------------------------------
        output = pl_module.model.encode(batch)

        emb = output["emb"]
        pixels_emb = output["pixels_emb"]
        act_emb = output["action_emb"]

        ctx_emb = emb[:, :self.ctx_len]
        ctx_pixels = pixels_emb[:, :self.ctx_len]
        ctx_act = act_emb[:, :self.ctx_len]

        pixels_dim = pixels_emb.size(-1)
        n_patches = pixels_emb.shape[2]

        # ------------------------------
        # Prediction with true actions
        # ------------------------------
        pred_true = pl_module.model.predict(
            ctx_emb,
        )
        pred_true = pred_true[..., :pixels_dim]

        # ------------------------------
        # Prediction with shuffled actions
        # ------------------------------
        perm = torch.randperm(ctx_act.shape[0], device=device)
        perm_act = ctx_act[perm]
        perm_act_tiled = repeat(
                perm_act.unsqueeze(2), 'b t 1 d -> b t p d', p=n_patches
            )
        
        # print(f"pixels_emb shape: {pixels_emb.shape}")
        # print(f"extra_tiled shape: {perm_act_tiled.shape}")
        perm_ctx_emb = torch.cat([ctx_pixels, perm_act_tiled], dim=3)

        pred_random = pl_module.model.predict(
            perm_ctx_emb,
        )
        pred_random = pred_random[..., :pixels_dim]

        emb_true = emb

        return emb_true, pred_true, pred_random


    @torch.no_grad()
    def on_train_batch_end(
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

        if self.model_struc == "jepa":
            emb_true, pred_true, pred_random = self.predict_jepa_emd(pl_module=pl_module, batch=batch, device=device)
        elif self.model_struc == "prejepa":
            emb_true, pred_true, pred_random = self.predict_prejepa_emd(pl_module=pl_module, batch=batch, device=device)

        mse = torch.mean((pred_true - pred_random) ** 2)

        trainer.logger.log_metrics(
            {
                "predictor/random_action_latent_mse": mse.item(),
            },
            step=trainer.global_step,
        )

        emb_norms = emb_true.norm(dim=-1)
        mean = emb_norms.mean()
        std  = emb_norms.std()
        min  = emb_norms.min()
        max  = emb_norms.max()

        z = emb_true.reshape(-1, emb_true.shape[-1])
        emb_variance = z.var(dim=0)

        mean_emb_var = emb_variance.mean()
        min_emb_var = emb_variance.min()

        trainer.logger.log_metrics(
            {
                "encoder/mean_latent": mean.item(),
                "encoder/std_latent": std.item(),
                "encoder/min_latent": min.item(),
                "encoder/max_latent": max.item(),
                "encoder/mean_emb_var": mean_emb_var.item(),
                "encoder/min_emb_var": min_emb_var.item(),
            },
            step=trainer.global_step,
        )

        pred_norms = pred_true.norm(dim=-1)

        # cos = F.cosine_similarity(
        #     pred_true.flatten(0,-2),
        #     emb_true[:, self.ctx_len:].flatten(0,-2),
        #     dim=-1,
        # ).mean()
        trainer.logger.log_metrics(
        {
            "predictor/mean_norm": pred_norms.mean().item(),
            "predictor/std_norm": pred_norms.std().item(),
        },
        step=trainer.global_step
        )






class SaveCkptCallback(Callback):
    """Callback to save model checkpoint after each epoch using save_pretrained."""

    def __init__(self, run_name, cfg, epoch_interval=1):
        super().__init__()
        self.run_name = run_name
        self.cfg = cfg
        self.epoch_interval = epoch_interval

    def on_train_epoch_end(self, trainer, pl_module):
        if not trainer.is_global_zero:
            return
        epoch = trainer.current_epoch + 1

        

        if epoch % self.epoch_interval == 0:
            self._save(pl_module.model, epoch)
        if epoch == trainer.max_epochs:
            self._save(pl_module.model, epoch)

    def _save(self, model, epoch):
        save_pretrained(
            model,
            run_name=self.run_name,
            config=self.cfg,
            filename=f'weights_epoch_{epoch}.pt',
        )
