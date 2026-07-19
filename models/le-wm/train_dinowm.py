import multiprocess
import multiprocessing as mp

import sys
sys.modules['multiprocessing'] = multiprocess

try:
    multiprocess.set_start_method('forkserver', force=True)
except RuntimeError:
    pass

import os
from functools import partial
from pathlib import Path
import numpy as np

import hydra
import lightning as pl
import stable_pretraining as spt
import stable_worldmodel as swm
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from datetime import datetime
import torch
from lightning.pytorch.loggers import WandbLogger, CSVLogger
from omegaconf import OmegaConf, open_dict
import os
from functools import partial
from pathlib import Path
import numpy as np

import hydra
import lightning as pl
import stable_pretraining as spt
import stable_worldmodel as swm
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from datetime import datetime
import torch
from lightning.pytorch.loggers import WandbLogger, CSVLogger
from omegaconf import OmegaConf, open_dict
from torch.nn import functional as F

from module import SIGReg
from utils import get_column_normalizer, get_img_preprocessor, instantiate_prejepa_model, \
            sanity_check_on_loaded_batch, collect_all_callbacks
from utils import ModelObjectCallBack, LinearProbeCallback

from preprocessing import depth_img_preprocessor, normal_img_preprocessor, rgbd_img_preprocessor, da3_img_preprocessor

from torch.utils.data import Subset
import random

import stable_worldmodel.data

# ---------------------------------------------------------------------------
# Forward
# ---------------------------------------------------------------------------


def _strip_action_dims(tensor, action_range):
    """Remove the action dimensions from the last axis."""
    return torch.cat(
        [tensor[..., : action_range[0]], tensor[..., action_range[1] :]],
        dim=-1,
    )


def dinowm_forward(self, batch, stage, cfg):
    """Encode observations, predict next states, compute losses."""
    batch['action'] = torch.nan_to_num(batch['action'], 0.0).squeeze()

    batch = self.model.encode(
        batch,
        target='emb',
        emb_keys=['action'],
        is_video=cfg.backbone.get('is_video_encoder', False),
    )

    embedding = batch['emb'][:, : cfg.wm.history_size, ...]
    pred_embedding = self.model.predict(embedding)
    target_embedding = batch['emb'][:, cfg.wm.num_preds :, ...].detach()

    # Per-modality losses
    pixels_dim = batch['pixels_emb'].size(-1)
    batch['pixels_loss'] = F.mse_loss(
        pred_embedding[..., :pixels_dim], target_embedding[..., :pixels_dim]
    )

    start, action_range = pixels_dim, [0, 0]
    for key in self.model.extra_encoders:
        dim = batch[f'{key}_emb'].size(-1)
        lo, hi = start, start + dim
        if key == 'action':
            action_range = [lo, hi]
        else:
            batch[f'{key}_loss'] = F.mse_loss(
                pred_embedding[..., lo:hi],
                target_embedding[..., lo:hi].detach(),
            )
        start = hi

    # Actionless embeddings (for probes and total loss)
    batch['actionless_emb'] = _strip_action_dims(batch['emb'], action_range)
    batch['actionless_prev_emb'] = _strip_action_dims(embedding, action_range)
    batch['actionless_pred_emb'] = _strip_action_dims(
        pred_embedding, action_range
    )
    batch['actionless_target_emb'] = _strip_action_dims(
        target_embedding, action_range
    )

    batch['loss'] = F.mse_loss(
        batch['actionless_pred_emb'],
        batch['actionless_target_emb'].detach(),
    )

    if batch['loss'].isnan():
        raise ValueError('NaN loss encountered!')

    self.log_dict(
        {f'{stage}/{k}': v.detach() for k, v in batch.items() if '_loss' in k},
        on_step=True,
        on_epoch=True,
        sync_dist=True,
    )
    return batch

@hydra.main(version_base=None, config_path="./config/train", config_name="prejepa")
def run(cfg):
    #########################
    ##       dataset       ##
    #########################

    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    dataset_name_without_full_path = cfg.data.dataset.name.split("/")[-1]
    SAVE_CKPT_PATH = f"/home/student/users/Public_workspace/le-wm-3DGeom/models/le-wm/outputs/checkpoints/{dataset_name_without_full_path}/"
    os.makedirs(SAVE_CKPT_PATH, exist_ok=True)
    RUN_OUTPUT_DIR=f"/home/student/users/Public_workspace/le-wm-3DGeom/models/le-wm/outputs/dinoresults_15_07/{dataset_name_without_full_path}/{run_timestamp}"
    os.makedirs(RUN_OUTPUT_DIR,exist_ok=True )

    DEPTH = "depth" in cfg.data.dataset.name.lower()
    NORMALS = "normals" in cfg.data.dataset.name.lower()
    RGB_DEPTH = "rgb_depth" in cfg.data.dataset.name.lower()
    DA3_DEPTH = "da3" in  cfg.data.dataset.name.lower()

    dataset = swm.data.HDF5Dataset(**cfg.data.dataset, transform=None)

    if (RGB_DEPTH):
        print("Initializing RGBD preprocessor\n")
        transforms = [rgbd_img_preprocessor(img_size=cfg.img_size,)]       # RGB+D
    elif (DA3_DEPTH): 
        print("Initializing DA3 preprocessor\n")
        transforms = [da3_img_preprocessor(img_size=cfg.img_size)]
    elif DEPTH: 
        print("Initializing DEPTH preprocessor\n")
        transforms = [depth_img_preprocessor(cfg.img_size)]                             # DEPTH
    elif NORMALS: 
        print("Initializing NORMS preprocessor\n")
        transforms = [normal_img_preprocessor(cfg.img_size)]                            # NORMALS
    else:
        print("Initializing RGB preprocessor\n")
        transforms = [get_img_preprocessor(source='pixels', target='pixels', img_size=cfg.img_size)]    # RGB
    
    logger = None
    if cfg.wandb.enabled:
        logger = WandbLogger(**cfg.wandb.config)
        logger.log_hyperparams(OmegaConf.to_container(cfg))


    print(dataset.clip_indices[:10])
    print(dataset.clip_indices[-10:])

    print("\n\nDataset initialized with the following parameters:")
    print(OmegaConf.to_yaml(cfg))
    
    with open_dict(cfg):
        for col in cfg.data.dataset.keys_to_load:

            if col in ["qpos", "qvel"]:
                continue

            if col.startswith("pixels"):
                continue

            normalizer = get_column_normalizer(dataset, col, col)
            transforms.append(normalizer)

            setattr(cfg.wm, f"{col}_dim", dataset.get_dim(col))

    transform = spt.data.transforms.Compose(*transforms)
    dataset.transform = transform

    print("dataset=", cfg.data.dataset)
    print("dataset.column_names=", dataset.column_names)
    print("dataset.frameskip=", dataset.frameskip)
    print("dataset.span=", dataset.span)
    print("dataset.num_steps=", dataset.num_steps)
    print("total clip_indices=", len(dataset.clip_indices))
    print("dataset samples= ", len(dataset))

    ##############################
    ##       DATALOADERS      ##
    ##############################
    rnd_gen = torch.Generator().manual_seed(cfg.seed)
    print("dataset samples=", len(dataset))
    train_loader = torch.utils.data.DataLoader(dataset, **cfg.loader, shuffle=True, drop_last=True, generator=rnd_gen)
    val_loader = train_loader
    print("number of batches in data loader=", len(train_loader))

    print("\n\nInspect a single sample from the dataset")
    ## SANITY CHECKS

    sample = dataset[0]
    for k, v in sample.items():
        if hasattr(v, "shape"):
            print(k, v.shape)
        else:
            print(k, type(v))
    img = sample["pixels"]
    print("img.shape=", img.shape)
    print("img.dtype=", img.dtype)
    print(img.min(), img.max())
    print(img.mean(), img.std())

    if DEPTH: 
        frame0 = sample["pixels"][0, 0].cpu().numpy()   # first frame, first channel

        # Undo [-1,1] normalization
        frame0 = (frame0 + 1.0) / 2.0
        frame0 = frame0 * (3.0 - 0.5) + 0.5

        plt.figure(figsize=(5,5))
        plt.imshow(frame0, cmap="viridis", vmin=0.5, vmax=3.0)
        plt.colorbar(label="Depth (m)")
        plt.axis("off")

        outfile = f"{RUN_OUTPUT_DIR}/depth_after_pipeline.png"
        plt.savefig(outfile, dpi=200, bbox_inches="tight")
        plt.close()


    print("\n\nInspect a full batch from the dataloader")
    batch = next(iter(train_loader))
    sanity_checks_output_dir = f"{RUN_OUTPUT_DIR}/sanity_checks"
    os.makedirs(sanity_checks_output_dir, exist_ok=True)
    sanity_check_on_loaded_batch(batch, depth=DEPTH, output_dir=sanity_checks_output_dir)


    all_episode_ids = torch.load("episode_order.pt")
    print(all_episode_ids[:10])

    train_episode_count = cfg.train_num_episodes

    # Fixed validation split:
    selected_train_episodes = set(all_episode_ids[0:train_episode_count])
    selected_val_episodes   = set(all_episode_ids[train_episode_count:train_episode_count+cfg.val_num_episodes])

    train_indices = [
        idx
        for idx, (local_ep, _) in enumerate(dataset.clip_indices)
        if dataset.episode_ids[local_ep] in selected_train_episodes
    ]

    val_indices = [
        idx
        for idx, (local_ep, _) in enumerate(dataset.clip_indices)
        if dataset.episode_ids[local_ep] in selected_val_episodes
    ]

    train_set = Subset(dataset, train_indices)
    val_set = Subset(dataset, val_indices)
    # train_set = dataset
    # val_set = dataset

    print(
        f"Dataset split | "
        f"train episodes={len(selected_train_episodes)} "
        f"train samples={len(train_set)} | "
        f"val episodes={len(selected_val_episodes)} "
        f"val samples={len(val_set)}"
    )
    print(train_indices[:20])
    print(train_indices[-20:])
    print(dataset.clip_indices[:20])

    train_loader = torch.utils.data.DataLoader(train_set,**cfg.loader,shuffle=True,drop_last=True,generator=rnd_gen,)

    # val_kwargs = dict(cfg.loader)
    # val_kwargs["persistent_workers"] = False
    # val_kwargs["prefetch_factor"] = None
    # val_kwargs["num_workers"] = 16
    val_loader = torch.utils.data.DataLoader(val_set,**cfg.loader,shuffle=False,drop_last=False,)

    print()
    print("len(train_loader)=", len(train_loader))
    print("len(val_loader)=", len(val_loader))

    ##############################
    ##       MODEL / OPTIM      ##
    ##############################

    world_model = instantiate_prejepa_model(cfg)

    optimizers = {
        'model_opt': {
            "modules": 'model',
            "optimizer": dict(cfg.optimizer),
            "scheduler": {"type": "LinearWarmupCosineAnnealingLR"},
            "interval": "epoch",
        },
    }

    data_module = spt.data.DataModule(train=train_loader, val=val_loader)
    world_model = spt.Module(
        model = world_model,
        forward=partial(dinowm_forward, cfg=cfg),
        optim=optimizers,
    )

    print(type(world_model))
    print(world_model.__class__)
    print(world_model.__class__.__module__)
    print(world_model.__class__.__mro__)

    ##########################
    ##       TRAINING       ##
    ##########################

    run_id = cfg.get("subdir") or ""
    run_dir = Path(swm.data.utils.get_cache_dir(), run_id)

    csv_logger = CSVLogger(save_dir="logs", name="csv")

    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.yaml", "w") as f:
        OmegaConf.save(cfg, f)

    all_callbacks = collect_all_callbacks(cfg, RUN_OUTPUT_DIR, model="prejepa")

    run_baseline_ckpt = cfg.get("run_baseline_ckpt", False)
    if run_baseline_ckpt:       # Evaluate the model on the official lewm weights

        # checkpoint_path = "/home/student/data/baseline_ckpt/hf_cube/weights.pt"
        # state_dict = torch.load(checkpoint_path, map_location="cpu")
        # world_model.model.load_state_dict(state_dict, strict=True)

        # print(f"\n\nLoaded pretrained weights into the model from {checkpoint_path}.")
        # print(f"\n\nCheckpoint found at {checkpoint_path}. Validating model with this checkpoint.")

        trainer = pl.Trainer(
            **cfg.trainer,
            callbacks=all_callbacks,
            num_sanity_val_steps=0,
            logger=logger,
            enable_checkpointing=False,
        )

        probe_callback = LinearProbeCallback()
        probe_callback.run_probe_call(train_loader=train_loader, val_loader=val_loader, trainer=trainer, pl_module=world_model)

        trainer.validate(
            model=world_model,
            datamodule=data_module,
            ckpt_path=None,
        )
        manager = spt.Manager(
            trainer=trainer,
            module=world_model,
            data=data_module,
            ckpt_path=None  # run_dir / f"{cfg.output_model_name}_weights.ckpt",
        )
        manager()

        print("\n\nValidation on baseline checkpoint complete. ")
        
        return
    else: 
        print("\n\nNo baseline checkpoint loaded. Starting training from scratch.\n\n")
        trainer = pl.Trainer(
            **cfg.trainer,
            callbacks=all_callbacks,
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

        print("train batches =", len(train_loader))
        print("val batches   =", len(val_loader))

        trainer = manager.trainer
        print("trainer.limit_train_batches =", trainer.limit_train_batches)
        print("trainer.limit_val_batches   =", trainer.limit_val_batches)
        print("trainer.fast_dev_run        =", trainer.fast_dev_run)
        print("trainer.overfit_batches     =", trainer.overfit_batches)

        manager()

        print("\n\n -- DONE WITH TRAINING -- \n\n")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        torch.save(world_model.model.state_dict(),os.path.join(SAVE_CKPT_PATH, f"weights_{timestamp}.pt"))
        print(f"Saved checkpoint: {SAVE_CKPT_PATH}/weights_{timestamp}.pt")
    
        return

if __name__ == "__main__":
    run()
