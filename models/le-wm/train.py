import os
from functools import partial
from pathlib import Path

import hydra
import lightning as pl
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from lightning.pytorch.loggers import WandbLogger, CSVLogger
from omegaconf import OmegaConf, open_dict

from jepa import JEPA
from module import ARPredictor, Embedder, MLP, SIGReg
from utils import get_column_normalizer, get_img_preprocessor, ModelObjectCallBack

from torch.utils.data import Subset
import random

from torch.profiler import profile, record_function, ProfilerActivity
from lightning.pytorch.profilers import PyTorchProfiler
from lightning.pytorch.callbacks import EarlyStopping


def lejepa_forward(self, batch, stage, cfg):
    """encode observations, predict next states, compute losses."""
    # print("*****************************************")
    # print("\n\n*** FORWARD PASS *** \n\n")
    # print(f"Forward pass at stage: {stage} and batch_idx: {batch['batch_idx']}")
    # print(f"Batch keys: {batch.keys()}")
    # print(f"Batch 'pixels' shape: {batch['pixels'].shape}")
    # print(f"Batch 'action' shape: {batch['action'].shape}")
    # print(f"Batch 'observation' shape: {batch['observation'].shape}")
    # print(f"Batch 'proprio' shape: {batch['proprio'].shape}")



    ctx_len = cfg.wm.history_size
    n_preds = cfg.wm.num_preds
    lambd = cfg.loss.sigreg.weight

    # Replace NaN values with 0 (occurs at sequence boundaries)
    with record_function("nan_to_num"):
        batch["action"] = torch.nan_to_num(
            batch["action"], 0.0
        )

    # batch["action"] = torch.nan_to_num(batch["action"], 0.0)
    # print(batch["action"][0])  # print the first action tensor after NaN replacement to verify
    # print("Encoding batch")
    with record_function("model_encode"):
            output = self.model.encode(batch)

    # output = self.model.encode(batch)           # [B,T,D]                           

    emb = output["emb"]  # (B, T, D)    -> T = number of sampled frames in one batch sequence, eg [f0, f4, f8, f12] with T=4 
    act_emb = output["act_emb"]

    # print("Encoded embeddings shape: ", emb.shape)
    # print("Encoded action embeddings shape: ", act_emb.shape)

    with record_function("slice_context"):
        ctx_emb = emb[:, :ctx_len]
        ctx_act = act_emb[:, :ctx_len]
        tgt_emb = emb[:, n_preds:].contiguous()

    # ctx_emb = emb[:, :ctx_len]
    # ctx_act = act_emb[:, : ctx_len]

    # print("Context embeddings shape: ", ctx_emb.shape)
    # print("Context action embeddings shape: ", ctx_act.shape)

    # tgt_emb = emb[:, n_preds:] # label

    # print("Target embeddings shape: ", tgt_emb.shape)
    # print("ctx_len: ", ctx_len, "n_preds: ", n_preds)

    # print("Running prediction")
    with record_function("model_predict"):
        pred_emb = self.model.predict(
            ctx_emb,
            ctx_act
        )
    # pred_emb = self.model.predict(ctx_emb, ctx_act) # pred
    # print("Prediction complete. pred_emb shape: ", pred_emb.shape)

    # LeWM loss
    # print("Target embedding shape: ", tgt_emb.shape)
    with record_function("pred_loss"):
        output["pred_loss"] = (
            (pred_emb - tgt_emb)
            .pow(2)
            .mean()
        )
    # output["pred_loss"] = (pred_emb - tgt_emb).pow(2).mean()

    with record_function("sigreg_loss"):
        output["sigreg_loss"] = self.sigreg(
            emb.transpose(0, 1)
    )
    # output["sigreg_loss"]= self.sigreg(emb.transpose(0, 1))
    with record_function("total_loss"):
        output["loss"] = (
            output["pred_loss"]
            + lambd * output["sigreg_loss"]
        )
        
    # output["loss"] = output["pred_loss"] + lambd * output["sigreg_loss"]  

    losses_dict = {f"{stage}/{k}": v.detach() for k, v in output.items() if "loss" in k}
    self.log_dict(losses_dict, on_step=False, on_epoch=True, sync_dist=True)

    # print("*****************************************")
    return output

@hydra.main(version_base=None, config_path="./config/train", config_name="lewm")
def run(cfg):
    #########################
    ##       dataset       ##
    #########################

    dataset = swm.data.HDF5Dataset(**cfg.data.dataset, transform=None, )
    transforms = [get_img_preprocessor(source='pixels', target='pixels', img_size=cfg.img_size)]

    print("\n\nDataset initialized with the following parameters:")
    print(OmegaConf.to_yaml(cfg))
    print(f"self.span={dataset.span}")
    
    with open_dict(cfg):
        for col in cfg.data.dataset.keys_to_load:
            if col.startswith("pixels"):
                continue

            normalizer = get_column_normalizer(dataset, col, col)
            transforms.append(normalizer)

            setattr(cfg.wm, f"{col}_dim", dataset.get_dim(col))

    transform = spt.data.transforms.Compose(*transforms)
    dataset.transform = transform

    # dataset.clip_indices = dataset.clip_indices[:64]  # for debugging, use only the first 1000 clips
    # episode_one = dataset.clip_indices[0:182]
    # episode_two = dataset.clip_indices[182:364]
    # dataset.clip_indices = [episode_one, episode_two]  # Use only the first two episodes for training and validation

    # ***************************     ORIGINAL DATA SPIT    ***************************
    rnd_gen = torch.Generator().manual_seed(cfg.seed)
    # train_set, val_set = spt.data.random_split(
    #     dataset, lengths=[cfg.train_split, 1 - cfg.train_split], generator=rnd_gen
    # )

    # train_eps = set()
    # val_eps = set()

    # for idx in train_set.indices: 
    #     ep_id = dataset.clip_indices[idx][0]
    #     train_eps.add(ep_id)

    # for idx in val_set.indices:
    #     ep_id = dataset.clip_indices[idx][0]
    #     val_eps.add(ep_id)

    # print("Overlap between train and val sets:", train_eps.intersection(val_eps))
    # ***************************************************************************************

    all_episode_ids = torch.load("episode_order.pt")
    print("Total unique episodes in dataset: ", len(all_episode_ids))
    print("First 20 episode IDs: ", all_episode_ids[:20])

    val_episode_count = cfg.val_num_episodes
    train_episode_count = cfg.train_num_episodes

    # fixed validation
    val_episodes = set(
        all_episode_ids[:val_episode_count]
    )

    train_pool = all_episode_ids[val_episode_count:]
    selected_train_episodes = set(
        train_pool[:train_episode_count]
    )

    # episodes → indices
    train_indices = [
        idx for idx, (ep_id, _) in enumerate(dataset.clip_indices)
        if ep_id in selected_train_episodes
    ]

    val_indices = [
        idx for idx, (ep_id, _) in enumerate(dataset.clip_indices)
        if ep_id in val_episodes
    ]

    train_set = Subset(dataset, train_indices)
    val_set = Subset(dataset, val_indices)
   
    print(f"Train episodes: {len(selected_train_episodes)}")
    print(f"Val episodes: {len(val_episodes)}")
    print(f"Train samples: {len(train_set)}")
    print(f"Val samples: {len(val_set)}")

    train = torch.utils.data.DataLoader(train_set, **cfg.loader,shuffle=True, drop_last=True, generator=rnd_gen)
    val = torch.utils.data.DataLoader(val_set, **cfg.loader, shuffle=False, drop_last=False)

    i = 0
    for batch in train:
        print(f"Batch keys: {batch.keys()}")
        print(f"Batch 'pixels' shape: {batch['pixels'].shape}")
        print(f"Batch 'action' shape: {batch['action'].shape}")
        i += 1
        if i == 2:
            break

        
    ##############################
    ##       model / optim      ##
    ##############################

    encoder = spt.backbone.utils.vit_hf(
        cfg.encoder_scale,
        patch_size=cfg.patch_size,
        image_size=cfg.img_size,
        pretrained=False,
        use_mask_token=False,
    )

    total = 0
    trainable = 0

    for name, p in encoder.named_parameters():
        total += p.numel()
        if p.requires_grad:
            trainable += p.numel()
            print(f"TRAINABLE: {name}")

    print(f"Total params: {total:,}")
    print(f"Trainable params: {trainable:,}")


    hidden_dim = encoder.config.hidden_size
    embed_dim = cfg.wm.get("embed_dim", hidden_dim)
    effective_act_dim = cfg.data.dataset.frameskip * cfg.wm.action_dim
    
    predictor = ARPredictor(
        num_frames=cfg.wm.history_size,
        input_dim=embed_dim,
        hidden_dim=hidden_dim,
        output_dim=hidden_dim,
        **cfg.predictor,
    )

    action_encoder = Embedder(input_dim=effective_act_dim, emb_dim=embed_dim)
    
    projector = MLP(
        input_dim=hidden_dim,
        output_dim=embed_dim,
        hidden_dim=2048,
        norm_fn=torch.nn.BatchNorm1d,
    )




    predictor_proj = MLP(
        input_dim=hidden_dim,
        output_dim=embed_dim,
        hidden_dim=2048,
        norm_fn=torch.nn.BatchNorm1d,
    )

    world_model = JEPA(
        encoder=encoder,
        predictor=predictor,
        action_encoder=action_encoder,
        projector=projector,
        pred_proj=predictor_proj,
    )

    optimizers = {
        'model_opt': {
            "modules": 'model',
            "optimizer": dict(cfg.optimizer),
            "scheduler": {"type": "LinearWarmupCosineAnnealingLR"},
            "interval": "epoch",
        },
    }

    data_module = spt.data.DataModule(train=train, val=val)
    world_model = spt.Module(
        model = world_model,
        sigreg = SIGReg(**cfg.loss.sigreg.kwargs),
        forward=partial(lejepa_forward, cfg=cfg),
        optim=optimizers,
    )

    print(type(world_model))
    print(world_model.__class__)
    print(world_model.__class__.__module__)
    print(world_model.__class__.__mro__)

    ##########################
    ##       training       ##
    ##########################

    run_id = cfg.get("subdir") or ""
    run_dir = Path(swm.data.utils.get_cache_dir(), run_id)

    logger = None
    if cfg.wandb.enabled:
        logger = WandbLogger(**cfg.wandb.config)
        logger.log_hyperparams(OmegaConf.to_container(cfg))

    csv_logger = CSVLogger(save_dir="logs", name="csv")

    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.yaml", "w") as f:
        OmegaConf.save(cfg, f)

    object_dump_callback = ModelObjectCallBack(
        dirpath=run_dir, filename=cfg.output_model_name, epoch_interval=1,
    )

    early_stop_callback = EarlyStopping(
        monitor="validate/loss",
        patience=cfg.early_stop_patience,
        mode="min",
        min_delta=cfg.early_stop_min_improv,
        verbose=True,
    )


    is_validation = cfg.get("run_validation", False)
    if is_validation:

        checkpoint_path = "/home/student/data/baseline_ckpt/hf_cube/weights.pt"
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        world_model.model.load_state_dict(state_dict, strict=True)
        print(f"\n\nLoaded pretrained weights into the model from {checkpoint_path}. Starting training with these weights.")

        ### Validate the model before training to check if everything is working correctly
        print(f"\n\nCheckpoint found at {checkpoint_path}. Validating model with this checkpoint before training.")

        trainer = pl.Trainer(
            **cfg.trainer,
            callbacks=[object_dump_callback, early_stop_callback],
            num_sanity_val_steps=0,
            logger=logger,
            enable_checkpointing=False,
        )

        trainer.validate(
            model=world_model,
            datamodule=data_module,
            ckpt_path=None,
        )

        print("\n\nValidation complete. ")
        
        trainer = pl.Trainer(
            **cfg.trainer,
            callbacks=[object_dump_callback],
            num_sanity_val_steps=1,
            logger= logger,                               #[logger, csv_logger],
            enable_checkpointing=True,
        )

        manager = spt.Manager(
            trainer=trainer,
            module=world_model,
            data=data_module,
            ckpt_path=None  # run_dir / f"{cfg.output_model_name}_weights.ckpt",
        )

        manager()
        return
    else: 
        print("\n\nNo checkpoint found. Starting training from scratch.\n\n")
        trainer = pl.Trainer(
            **cfg.trainer,
            callbacks=[object_dump_callback, early_stop_callback],
            num_sanity_val_steps=0,
            logger=logger,
            enable_checkpointing=False,
        )

        # trainer.validate(
        #     model=world_model,
        #     datamodule=data_module,
        #     ckpt_path=None,
        # )

        manager = spt.Manager(
            trainer=trainer,
            module=world_model,
            data=data_module,
            ckpt_path=None  # run_dir / f"{cfg.output_model_name}_weights.ckpt",
        )

        manager()
        return

if __name__ == "__main__":
    run()
