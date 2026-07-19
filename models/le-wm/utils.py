import numpy as np
import torch
from pathlib import Path
from stable_pretraining import data as dt
from lightning.pytorch.callbacks import Callback
from custom_callbacks import LeWMPredictorIGCallback, LinearProbeCallback, RandomActionLatentMSECallback
from lightning.pytorch.callbacks import EarlyStopping

import os

import numpy as np
from PIL import Image

import stable_pretraining as spt
from stable_worldmodel.wm.prejepa.module import create_backbone, CausalPredictor
from stable_worldmodel.wm.prejepa.module import Embedder as Prejepa_Embedder
from stable_worldmodel.wm.prejepa import PreJEPA
from jepa import JEPA
from module import ARPredictor, Embedder, MLP, SIGReg
import torchvision.transforms as T
import torch
import torch.nn as nn
import time


def get_img_preprocessor(source: str, target: str, img_size: int = 224):
    imagenet_stats = dt.dataset_stats.ImageNet
    to_image = dt.transforms.ToImage(
        **imagenet_stats,
        source=source,
        target=target,
    )
    resize = dt.transforms.Resize(img_size, source=source, target=target)
    return dt.transforms.Compose(to_image, resize)


def get_column_normalizer(dataset, source: str, target: str):
    """Get normalizer for a specific column in the dataset."""
    col_data = dataset.get_col_data(source)
    data = torch.from_numpy(np.array(col_data))
    data = data[~torch.isnan(data).any(dim=1)]
    mean = data.mean(0, keepdim=True).clone()
    std = data.std(0, keepdim=True).clone()

    def norm_fn(x):
        return ((x - mean) / std).float()

    normalizer = dt.transforms.WrapTorchTransform(norm_fn, source=source, target=target)
    return normalizer


def sanity_check_on_loaded_batch(batch, depth, output_dir, num_sanity_checks=5):
    print("\nRunning sanity check on batch from data loader, saving outputs to: ", output_dir)
 
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    batch_sanity_checks = 0
    
    print(batch["pixels"].shape)

    for idx, batch_item_pixels in enumerate(batch["pixels"]):
        for i, frame in enumerate(batch_item_pixels):

            arr = frame.numpy()
            arr = np.transpose(arr, (1,2,0))

            if idx == 0 and i == 0: 
                print("frame.shape=", arr.shape)
                print("frame.dtype=", arr.dtype)
                print("frame.min and max=", arr.min(), arr.max())

            if depth: 
                arr = frame.numpy()
                
                # If shape is (1, H, W), squeeze channel
                if arr.ndim == 3:
                    arr = arr[0]  # (H, W)
                
                if idx == 0 and i == 0:
                    print("frame.shape=", arr.shape)
                    print("frame.dtype=", arr.dtype)
                    print("frame.min and max=", arr.min(), arr.max())
                
                # Denormalize from [-1, 1] back to meters [0.5, 3.0]
                arr = (arr + 1) / 2  # [0, 1]
                arr = (arr * 2.5) + 0.5  # [0.5, 3.0]
                
                # Normalize to 0-255 for saving as image
                arr = (arr - 0.5) / 2.5  # Back to [0, 1]
                arr = (arr * 255).clip(0, 255).astype(np.uint8)
                
                # Save as grayscale or colormap
                # im = Image.fromarray(arr, mode='L')  # 'L' for grayscale
                # OR use colormap:
                import matplotlib.cm as cm
                arr_colormap = (cm.viridis(arr / 255)[:, :, :3] * 255).astype(np.uint8)
                im = Image.fromarray(arr_colormap)
            else: 
                arr = arr * std + mean
                arr = (arr * 255).clip(0, 255).astype(np.uint8)
                im = Image.fromarray(arr)
            
            im.save(os.path.join(output_dir, f'loader_batch_{idx}_frame_{i}.png'))
        batch_sanity_checks += 1
        if batch_sanity_checks >= num_sanity_checks:    
            break           # break from within this batch



class ModelObjectCallBack(Callback):
    """Callback to pickle model object after each epoch."""

    def __init__(self, dirpath, filename="model_object", epoch_interval: int = 1):
        super().__init__()
        self.dirpath = Path(dirpath)
        self.filename = filename
        self.epoch_interval = 1

    def on_train_epoch_end(self, trainer, pl_module):
        super().on_train_epoch_end(trainer, pl_module)

        output_path = (
            self.dirpath
            / f"{self.filename}_epoch_{trainer.current_epoch + 1}_object.ckpt"
        )

        print("Saving ckpt to ", output_path)

        if trainer.is_global_zero:
            if (trainer.current_epoch + 1) % self.epoch_interval == 0:
                self._dump_model(pl_module.model, output_path)

                print("\n\nCKPT SAVED!\n\n")

            # save final epoch
            if (trainer.current_epoch + 1) == trainer.max_epochs:
                self._dump_model(pl_module.model, output_path)

    def _dump_model(self, model, path):
        try:
            torch.save(model, path)
        except Exception as e:
            print(f"Error saving model object: {e}")





def instantiate_world_model(cfg):

    if cfg.concat_RGBD:
        print("CONCATENATING RGB+D")
        num_vit_channels = 4
    else: 
        num_vit_channels = 3

    encoder = spt.backbone.utils.vit_hf(
        cfg.encoder_scale,
        patch_size=cfg.patch_size,
        image_size=cfg.img_size,
        pretrained=False,
        use_mask_token=False,
        num_channels=num_vit_channels
    )

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

    action_encoder = Embedder(
        input_dim=effective_act_dim,
        emb_dim=embed_dim,
    )

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

    return world_model


def instantiate_prejepa_model(cfg):

    # Instantiate a pretrained backbone as the encoder, freeze its weights
    # without the option for a video encoder or CNN encoder
    encoder = create_backbone(cfg.backbone.name)
    encoder.eval()
    encoder.requires_grad_(False)

    embed_dim = encoder.config.hidden_size + cfg.wm.action_encoding
    num_patches = (cfg.img_size // cfg.patch_size) ** 2
    effective_act_dim = cfg.data.dataset.frameskip * cfg.wm.action_dim

    predictor = CausalPredictor(
        num_patches=num_patches,
        num_frames=cfg.wm.history_size,
        dim=embed_dim,
        **cfg.predictor,
    )

    action_encoder = Prejepa_Embedder(
        in_chans=effective_act_dim,
        emb_dim=int(cfg.wm.action_encoding)
    )

    extra_encoders = nn.ModuleDict({
        "action": action_encoder,
    })

    world_model = PreJEPA(
        history_size=cfg.wm.history_size,
        num_pred=cfg.wm.num_preds,
        interpolate_pos_encoding=cfg.backbone.interpolate_pos_encoding,
        encoder=encoder,
        predictor=predictor,
        extra_encoders=extra_encoders,
    )

    return world_model

    



def collect_all_callbacks(cfg, run_dir, model="jepa"): 

    all_callbacks = []
    if cfg.train_epochs.get("save_ckpt_every_epochs", 0) != 0: 
        N_epochs = cfg.get("save_ckpt_every_epochs")
        print("\n\nSaving checkpoint every N epochs! N=", N_epochs )
        object_dump_callback = ModelObjectCallBack(
            dirpath=run_dir, filename=cfg.output_model_name, epoch_interval=N_epochs,
        )
        all_callbacks.append(object_dump_callback)

    if cfg.train_epochs.get("early_stop_active", False): 
        print("\nRunning with EarlyStopping !")
        early_stop_callback = EarlyStopping(
            monitor="validate/loss",
            patience=cfg.train_epochs.early_stop_patience,
            mode="min",
            min_delta=cfg.train_epochs.early_stop_min_improv,
            verbose=True,
        )
        all_callbacks.append(early_stop_callback)


    if cfg.train_epochs.get("integrated_gradients_active", False):
        print("\nIntegratedGradients Callback active")
        predictor_callback = LeWMPredictorIGCallback(every_n_epochs=1, n_steps=30, ctx_len=1)
        all_callbacks.append(predictor_callback)
    

    if cfg.train_epochs.get("linear_probing_active", False):
        print("\nLinear Probing Callback active")
        probing_callback = LinearProbeCallback(every_n_epochs=1, max_train_batches=150, max_val_batches=15, model=model)
        all_callbacks.append(probing_callback)


    if cfg.train_epochs.get("rnd_action_latent_mse", False):
        print("\RandomActionLatentMSECallback active")
        random_action = RandomActionLatentMSECallback(every_n_epochs=1, ctx_len=cfg.wm.history_size, model=model)
        all_callbacks.append(random_action)

    print("all callbacks=")
    print(all_callbacks)

    return all_callbacks