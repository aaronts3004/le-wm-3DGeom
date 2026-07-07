import os
from functools import partial
from pathlib import Path

import hydra
import lightning as pl
import stable_pretraining as spt
import stable_worldmodel as swm
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
from lightning.pytorch.loggers import WandbLogger, CSVLogger
from omegaconf import OmegaConf, open_dict

from jepa import JEPA
from module import ARPredictor, Embedder, MLP, SIGReg
from utils import get_column_normalizer, get_img_preprocessor, ModelObjectCallBack

from torch.utils.data import Subset
import torch.nn.functional as F
import random
os.environ["MUJOCO_GL"] = "egl"

from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint


OUTPUT_DIR = "outputs/overfit_one_sample/"
os.makedirs(OUTPUT_DIR, exist_ok=True)


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
    batch["action"] = torch.nan_to_num(batch["action"], 0.0)
    # print(batch["action"][0])  # print the first action tensor after NaN replacement to verify
    # print("Encoding batch")
    output = self.model.encode(batch)           # [B,T,D]                           

    emb = output["emb"]  # (B, T, D)    -> T = number of sampled frames in one batch sequence, eg [f0, f4, f8, f12] with T=4 
    act_emb = output["act_emb"]

    # print("Encoded embeddings shape: ", emb.shape)
    # print("Encoded action embeddings shape: ", act_emb.shape)

    ctx_emb = emb[:, :ctx_len]
    ctx_act = act_emb[:, :ctx_len]
    tgt_emb = emb[:, n_preds:].contiguous()

    # ctx_emb = emb[:, :ctx_len]
    # ctx_act = act_emb[:, : ctx_len]

    # print("Context embeddings shape: ", ctx_emb.shape)
    # print("Context action embeddings shape: ", ctx_act.shape)

    tgt_emb = emb[:, n_preds:] # label

    # print("Target embeddings shape: ", tgt_emb.shape)
    # print("ctx_len: ", ctx_len, "n_preds: ", n_preds)

    # print("Running prediction")
    pred_emb = self.model.predict(ctx_emb, ctx_act) # pred
    # print("Prediction complete. pred_emb shape: ", pred_emb.shape)

    # LeWM loss
    # print("Target embedding shape: ", tgt_emb.shape)
    output["pred_loss"] = (pred_emb - tgt_emb).pow(2).mean()
    output["sigreg_loss"]= self.sigreg(emb.transpose(0, 1))
    output["loss"] = output["pred_loss"] + lambd * output["sigreg_loss"]  

    losses_dict = {f"{stage}/{k}": v.detach() for k, v in output.items() if "loss" in k}
    self.log_dict(losses_dict, on_step=False, on_epoch=True, sync_dist=True)

    # print("*****************************************")
    return output

@hydra.main(version_base=None, config_path="./config/train", config_name="lewm")
def run(cfg):
    #########################
    ##       dataset       ##
    #########################

    import os
    from stable_worldmodel.data.utils import get_cache_dir

    dataset = swm.data.HDF5Dataset("/home/student/data/ogbench/cube_single_expert", transform=None, )
    transforms = [get_img_preprocessor(source='pixels', target='pixels', img_size=cfg.img_size)]

    print("\n\nDataset initialized with the following parameters:")
    print(OmegaConf.to_yaml(cfg))
    print(f"self.span={dataset.span}")
    print(dataset.h5_path)
    
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

    print(dataset.column_names)


    # ***************************     ORIGINAL DATA SPIT    ***************************
    rnd_gen = torch.Generator().manual_seed(cfg.seed)
    train_set, val_set = spt.data.random_split(
        dataset, lengths=[cfg.train_split, 1 - cfg.train_split], generator=rnd_gen
    )
    ################################################################################     DATASET: OVERFIT ONE ACTION  
    sample = dataset[0]
    for k, v in sample.items():
        if hasattr(v, "shape"):
            print(k, v.shape)
        else:
            print(k, type(v))

    sample_index=150
    overfit_subset = torch.utils.data.Subset(
        dataset,
        [sample_index] 
    )
    train = torch.utils.data.DataLoader(
        overfit_subset,
        batch_size=1,
        shuffle=False,
    )
    overfit_val_subset = train

    val = train

    print(f"Train samples: {len(train)}")
    print(f"Val samples: {len(val)}")

    batch = next(iter(train))
    print(batch.keys())

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    print("Inspect all batches")
    for batch in train:
        print(batch["pixels"].shape)

        for idx, batch_item_pixels in enumerate(batch["pixels"]):
            for i, frame in enumerate(batch_item_pixels):
                arr = frame.numpy()
                arr = np.transpose(arr, (1,2,0))
                arr_vis = arr * std + mean
                arr_vis = np.clip(arr_vis, 0, 1)
                arr_vis = (255 * arr_vis).astype(np.uint8)
                im = Image.fromarray(arr_vis)
                im.save(
                    os.path.join(OUTPUT_DIR, 
                    f'pixels_input_sample_{sample_index}_batch_{idx}_frame_{i}.png'
                    )
                )
            batch_item_pixels =  batch["pixels"]

        for idx, batch_item_qpos in enumerate(batch["qpos"]):
            print(f"qpos_{idx}=", batch_item_qpos)

        for idx, batch_item_qvel in enumerate(batch["qvel"]):
            print(f"qvel_{idx}=", batch_item_qvel)

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
            # print(f"TRAINABLE: {name}")

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
        norm_fn=torch.nn.LayerNorm,
    )

    predictor_proj = MLP(
        input_dim=hidden_dim,
        output_dim=embed_dim,
        hidden_dim=2048,
        norm_fn=torch.nn.LayerNorm,
    )

    world_model_JEPA = JEPA(
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
        model = world_model_JEPA,
        sigreg = SIGReg(**cfg.loss.sigreg.kwargs),
        forward=partial(lejepa_forward, cfg=cfg),
        optim=optimizers,
    )

    ##########################
    ##       LOAD CHECKPOINT       ##
    ##########################

    run_id = cfg.get("subdir") or ""
    run_dir = Path(swm.data.utils.get_cache_dir(), run_id)

    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.yaml", "w") as f:
        OmegaConf.save(cfg, f)


    ######################### 
    ### Action Sampling ### 
    #########################
    '''
    #                                                                                                                      MANUAL RANDOM ACTION SAMPLING 
    # Get first train sample from data loader
    world_model_JEPA.eval()
    sample_info = next(iter(train))

    # Sanity checks for shapes + save images as RGB
    print("sample_info['pixels'].shape=", sample_info["pixels"].shape)
    print("sample_info['action'].shape=", sample_info["action"].shape)

    sample_batch = sample_info["pixels"][0]
    frame0 = sample_batch[0]
    frame1 = sample_batch[1]

    print(sample_info["action"][0,0].shape)
    print(sample_info["action"][0,1].shape)
    print(sample_info["action"][0,0])
    print(sample_info["action"][0,1])

    arr = frame0.numpy()
    arr = np.transpose(arr, (1,2,0))
    arr_vis = arr * std + mean
    arr_vis = np.clip(arr_vis, 0, 1)
    arr_vis = (255 * arr_vis).astype(np.uint8)
    im = Image.fromarray(arr_vis)
    im.save(f'outputs/overfit/CEM/sanity_check_frameskip5_frame0_epsample_{sample_index}.png')

    arr = frame1.numpy()
    arr = np.transpose(arr, (1,2,0))
    arr_vis = arr * std + mean
    arr_vis = np.clip(arr_vis, 0, 1)
    arr_vis = (255 * arr_vis).astype(np.uint8)
    im = Image.fromarray(arr_vis)
    im.save(f'outputs/overfit/CEM/sanity_check_frameskip5_frame1_epsample_{sample_index}.png')

    
    # Run the encoder for frame0 + actions=[a0,a1,a2,a3,a4] + frame5 -> returns the latent embedding
    sample_emb = world_model_JEPA.encode(sample_info)

    pix_emb = sample_emb["emb"]                 
    act_emb = sample_emb["act_emb"] 
    print("pix_emb.shape=", pix_emb.shape)
    print("act_emb.shape=", act_emb.shape)

    # Run predictor on learned embeddings
    ctx_emb = pix_emb[:, :1]
    ctx_act = act_emb[:, :1]
    world_model_predicted_emb = world_model_JEPA.predict(ctx_emb, ctx_act)
    print("world_model_predicted_emb.shape=", world_model_predicted_emb.shape)

    # Compare prediction embedding against ground-truth target embedding
    target = pix_emb[:, 1:2]
    print("target.shape=", target.shape)

    true_error = F.mse_loss(world_model_predicted_emb, target)
    print("true error loss=", true_error)


    # Sample random actions and re-compute the embedding with this new random action, see if it provides an embedding that is closer to the ground-truth embedding
    print("\n\n Replace the action embedding with a random action")

    random_action = torch.randn_like(sample_info["action"][:, :1])
    random_act_emb = world_model_JEPA.action_encoder(random_action)
    print("random_act_emb.shape=", random_act_emb.shape)
    world_model_pred_random_emb = world_model_JEPA.predict(ctx_emb,random_act_emb)
    print("world_model_pred_random_emb.shape=", world_model_pred_random_emb.shape)
    random_error = F.mse_loss(world_model_pred_random_emb,target)
    print(f"random error loss: ", random_error)

    print(
        "pred_true - pred_random: ", 
        torch.norm(
            world_model_predicted_emb - world_model_pred_random_emb
        )
    )

    print("\n\n Replace the action embedding with a shuffled action")
    perm = torch.randperm(25)
    shuffled_action = sample_info["action"].clone()
    shuffled_action[..., :] = shuffled_action[..., perm]
    shuffled_act_emb = world_model_JEPA.action_encoder(shuffled_action)
    world_model_pred_shuffled_emb = world_model_JEPA.predict(ctx_emb,shuffled_act_emb)
    random_error = F.mse_loss(world_model_pred_shuffled_emb,target)
    print(f"shuffled error loss: ", random_error)
    print(
        "pred_true - pred_shuffled: ", 
        torch.norm(
            world_model_predicted_emb - world_model_pred_shuffled_emb
        )
    )

    print("\n\n Replace the action embedding with a gaussian perturbation")
    true_action = sample_info["action"][:,:1]
    for sigma in [0.01, 0.1, 0.5, 1.0]:
        random_action = true_action + sigma * torch.randn_like(true_action)
        random_act_emb = world_model_JEPA.action_encoder(random_action)
        print("gauss_act_emb.shape=", random_act_emb.shape)
        world_model_pred_random_emb = world_model_JEPA.predict(ctx_emb,random_act_emb)
        print("world_model_pred_random_emb.shape=", world_model_pred_random_emb.shape)
        random_error = F.mse_loss(world_model_pred_random_emb,target)
        print(f"sigma={sigma}, random error loss: ", random_error)

        print(
            "pred_true - pred_random: ", 
            torch.norm(
                world_model_predicted_emb - world_model_pred_random_emb
            )
        )

    print("\n\n Setup an action-based cost model")
    print("\n\n Try out 1000 in-distribution random actions and pick the best one")
    z0 = pix_emb[:, :1]                 # src emb
    z_goal = pix_emb[:, 1:2]            # tgt emb
    z_pred = world_model_JEPA.predict(z0,act_emb[:, :1])

    cost_true = F.mse_loss(z_pred,z_goal)
    print("true cost=", cost_true)

    actions = []
    costs = []
    rng = np.random.default_rng(cfg.seed)
    for i in range(1000):
        random_idx = rng.integers(low=0, high=1000)
        random_action = dataset[random_idx]["action"][:1]                       
        random_action = torch.randn_like(sample_info["action"][:, :1])
        random_act_emb = world_model_JEPA.action_encoder(random_action)
        pred = world_model_JEPA.predict(z0,random_act_emb)
        cost = F.mse_loss(pred,z_goal).item()
        costs.append(cost)
        actions.append(random_action.cpu().numpy())

    costs = np.array(costs)
    print("true:", cost_true.item())
    print("mean:", costs.mean())
    print("min:", costs.min())
    print("max:", costs.max())

    rank = np.sum(costs < cost_true.item())
    print("rank res=", rank)

    percentile = (
        np.sum(costs > cost_true.item())
        / len(costs)
    )
    print("percentile res=", percentile)

    plt.close()
    plt.hist(costs, bins=50)
    plt.axvline(cost_true.item())
    plt.savefig(f"outputs/overfit/CEM/cost_hist_in_distribution_frameskip_5_epsample_{sample_index}.png")

    best_idx = np.argsort(costs)[:10]
    print("best_idx=", best_idx)

    print("\n")
    for i in best_idx: 
        print("cost and actions for best action i=", i)
        print(costs[i])
        print(actions[i])
        print("---")
    print("\n")
    true_action = sample_info["action"][:, :1]
    best_action = actions[best_idx[0]]
    best_to_true_norm = np.linalg.norm(
        best_action - true_action.cpu().numpy()
    )
    print("best_to_true_norm=", best_to_true_norm)
    print("sample_qpos=", sample_info["qpos"])
    print("sample_qvel=", sample_info["qvel"])


    action_true = true_action.reshape(5, 5)
    print("action_true=", action_true)


    action_best = best_action.reshape(5, 5)
    print("action_best=", action_best)

    '''

    '''                                                try out different scale perturbations
    for scale in [0, 1, 10, 100]:

        random_action = scale * torch.randn_like(sample_info["action"][:, :1])
        random_act_emb = world_model_JEPA.action_encoder(random_action)
        print("random_act_emb.shape=", random_act_emb.shape)
        world_model_pred_random_emb = world_model_JEPA.predict(ctx_emb,random_act_emb)
        print("world_model_pred_random_emb.shape=", world_model_pred_random_emb.shape)
        random_error = F.mse_loss(world_model_pred_random_emb,target)
        print(f"scale={scale}, random error loss: ", random_error)

        print(
            "pred_true - pred_random: ", 
            torch.norm(
                world_model_predicted_emb - world_model_pred_random_emb
            )
        )
    '''

    #########################################################
    #                                                                   CEM OVERFIT EXPERIMENT
    #########################################################

    
    from types import SimpleNamespace
    from stable_worldmodel.solver import CEMSolver
    import gymnasium as gym

    import gymnasium
    import ogbench
    import mujoco
    import h5py

    CEM_OUTPUT_DIR=f"outputs/overfit/CEM/results_{sample_index}/"
    os.makedirs(CEM_OUTPUT_DIR,exist_ok=True )

    # -------------------------------------------------------
    # 1. Build source / goal inputs
    # -------------------------------------------------------
    world_model_JEPA.eval()
    world_model_JEPA.to("cuda")
    print(type(world_model_JEPA))
    print(hasattr(world_model_JEPA, "get_cost"))
    print(next(world_model_JEPA.parameters()).device)

    sample_info = next(iter(train))
    print("sample_info.keys()=", sample_info.keys())
    source_pixels = sample_info["pixels"][:, 0:1].cuda()
    goal_pixels   = sample_info["pixels"][:, 1:2].cuda()

    info_dict = {
        "pixels": source_pixels,
        "goal": goal_pixels,
    }

    # -------------------------------------------------------
    # 2. Configure CEM
    # -------------------------------------------------------
    
    solver = CEMSolver(
        model=world_model_JEPA,      # your trained world model
        batch_size=1,
        num_samples=1000,
        topk=50,
        n_steps=10,
        device="cuda",
    )
    env = gym.make(
        "visual-cube-single-v0",
        terminate_at_goal=False,
        mode="data_collection",
    )

    env.reset()
    solver.configure(
        action_space=env.action_space,
        n_envs=1,
        config=SimpleNamespace(
            horizon=1,      # IMPORTANT
            action_block=5, # frameskip
        ),
    )

    # -------------------------------------------------------
    # 3. Run CEM
    # -------------------------------------------------------

    info_dict["action"] = torch.zeros(
        1,      # batch
        1,      # history length
        25,     # action block
        device="cuda"
    )

    outputs = solver.solve(info_dict)
    cem_action = outputs["actions"]
    print("solver finished\n\n")
    print("cem_action.shape =", cem_action.shape)
    cem_action = cem_action[0, 0].numpy()

    # -------------------------------------------------------
    # 4. Compare to true action
    # -------------------------------------------------------

    true_action = sample_info["action"][0, 0].cpu().numpy()
    print(
        "cem_action_to_true_action_norm =",
        np.linalg.norm(cem_action - true_action)
    )
    # -------------------------------------------------------
    # 5. Rollout in OGBench
    # -------------------------------------------------------

    source_qpos = sample_info["qpos"][0, 0].numpy()
    source_qvel = sample_info["qvel"][0, 0].numpy()

    target_qpos = sample_info["qpos"][0, 1].numpy()
    target_qvel = sample_info["qvel"][0, 1].numpy()

    print("\n\n")
    print("dataset.sample_info.qpos.shape=", sample_info["qpos"].shape)
    print("dataset.sample_info.qpos=", sample_info["qpos"])
    print("dataset.sample_info.qvel.shape=", sample_info["qvel"].shape)
    print("dataset.sample_info.qvel=", sample_info["qvel"])

    print("\n\n")

    env.reset()
    env.unwrapped.set_state(
        source_qpos,
        source_qvel
    )
    img = env.unwrapped.render(camera="front_pixels")
    Image.fromarray(img).save(
        os.path.join(CEM_OUTPUT_DIR, f"first_source_state.png")
    )

    action_seq = cem_action.reshape(5, 5)
    print("CEM action seq=", action_seq)

    for a in action_seq:
        env.step(a)

    final_qpos = env.unwrapped.data.qpos.copy()
    final_qvel = env.unwrapped.data.qvel.copy()

    print(
        "final_cem_qpos_error =",
        np.linalg.norm(final_qpos - target_qpos)
    )

    print(
        "final_cem_qvel_error =",
        np.linalg.norm(final_qvel - target_qvel)
    )

    ####################################################### OGB visualizations 
    ### Visualize source f0
    env.reset()
    env.unwrapped.set_state(
        source_qpos,
        source_qvel
    )
    img = env.unwrapped.render(camera="front_pixels")
    Image.fromarray(img).save(
        os.path.join(CEM_OUTPUT_DIR, f"source.png")
    )

    ### Visualize target
    env.unwrapped.set_state(
        target_qpos,
        target_qvel
    )
    img = env.unwrapped.render(camera="front_pixels")
    Image.fromarray(img).save(
        os.path.join(CEM_OUTPUT_DIR, f"target.png")
    )

    ### Reset to source
    env.unwrapped.set_state(
        source_qpos,
        source_qvel
    )

    print("cem_action.shape =", cem_action.shape)
    cem_actions = cem_action.reshape(5, 5)
    print("cem_actions.shape =", cem_actions.shape)

    ### Rollout the true action 
    true_action = sample_info["action"][:,:1]

    true_action_reshaped = true_action.reshape(5, 5)
    print("TRUE ACTION=", true_action_reshaped)

    for i, a in enumerate(true_action):
        env.step(a)
        env.unwrapped.render(camera="front_pixels")
        Image.fromarray(img).save(
            os.path.join(CEM_OUTPUT_DIR,f"true_action_{i}.png")
        )

    ### Reset to source 
    env.unwrapped.set_state(
        source_qpos,
        source_qvel
    )
    for a in cem_action.reshape(5,5):
        env.step(a)
        env.unwrapped.render(camera="front_pixels")
        Image.fromarray(img).save(
            os.path.join(CEM_OUTPUT_DIR,f"cem_action_{i}.png")
        )


    return

if __name__ == "__main__":
    run()
