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
from utils import get_column_normalizer, get_img_preprocessor, ModelObjectCallBack, depth_img_preprocessor, instantiate_world_model

from datetime import datetime
from outputs import visualize_cem_solver

from torch.utils.data import Subset
import torch.nn.functional as F
import random
os.environ["MUJOCO_GL"] = "egl"

from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint


### For now we assume same source (f0) for all experiments
SRC_QPOS_0 = np.array([-1.8570e+00, -1.5349e+00,  2.1462e+00, -2.2110e+00, -1.5708e+00,
            2.7199e+00,  4.4934e-01,  5.8575e-04,  4.4669e-01, -4.4292e-01,
            4.4929e-01,  2.9956e-04,  4.4632e-01, -4.3992e-01,  4.7053e-01,
            4.2916e-03,  1.0841e-01, -7.5304e-01, -1.4355e-02,  1.7852e-02,
            6.5758e-01])

SRC_QVEL_0 = np.array([ 7.3992e-02,  4.4032e-01,  4.0368e-01, -6.6603e-01,  1.0991e-03,
            1.5137e-01, -4.6606e-04,  1.4755e-04,  6.1730e-04, -1.4372e-03,
            -7.2293e-04, -7.0954e-04, -2.5147e-03,  6.0789e-03, -5.2453e-02,
            4.6213e-02, -2.9620e-01, -1.9350e-01,  7.8925e-02, -7.7470e-02])


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


    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ##### ADJUST ACCORDINGLY #####
    CKPT_PATH = "/home/student/users/Aaron_workspace/le-wm-3DGeom/models/le-wm/outputs/checkpoints/ogbench/ogbench/cube_overfit_1_sources_1000_traj_1_actionBlocks_RGB/weights_20260701_123855.pt"
    # CKPT_PATH = "/home/student/users/Aaron_workspace/le-wm-3DGeom/models/le-wm/outputs/checkpoints/ogbench/ogbench/cube_overfit_1_sources_1000_traj_1_actionBlocks_DEPTH_inverse/weights_20260701_154317.pt"
    CKPT_TIMESTAMP = CKPT_PATH.split("/")[-1]
    print(f"Using checkpoint from timestamp: {CKPT_TIMESTAMP}") 

    DEPTH = False
    LINEAR=False 
    INVERSE=False
    if "DEPTH" in cfg.data.dataset.name:
        print("\n\nEVALUATING WITH DEPTH\n\n")
        DEPTH = True   
        if "linear" in cfg.data.dataset.name:
            print("\n\nEVALUATING WITH LINEAR NORMALIZATION\n\n")
            LINEAR = True
        elif "inverse" in cfg.data.dataset.name:
            print("\n\nEVALUATING WITH INVERSE NORMALIZATION\n\n")
            INVERSE = True
        else: 
            print("Uknonw DEPTH normalization type. Please CHECK again.")
            exit(1)

    CEM_OUTPUT_DIR=f"/home/student/users/Aaron_workspace/le-wm-3DGeom/models/le-wm/outputs/overfit/presentation_09_07/{cfg.data.dataset.name}/{CKPT_TIMESTAMP}/run_at_{run_timestamp}"
    os.makedirs(CEM_OUTPUT_DIR,exist_ok=True )

    dataset = swm.data.HDF5Dataset(**cfg.data.dataset, transform=None )
    if DEPTH: 
        transforms = [depth_img_preprocessor(cfg.img_size)]
    else:
        transforms = [get_img_preprocessor(source='pixels', target='pixels', img_size=cfg.img_size, depth=DEPTH)]

    print("\n\nDataset initialized with the following parameters:")
    print(OmegaConf.to_yaml(cfg))
    print(f"self.span={dataset.span}")
    print("dataset=", cfg.data.dataset)
    
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

    print("dataset.column_names=", dataset.column_names)
    print("dataset.frameskip=", dataset.frameskip)
    print("dataset.span=", dataset.span)
    print("dataset.num_steps=", dataset.num_steps)
    print("total clip_indices=", len(dataset.clip_indices))

    rnd_gen = torch.Generator().manual_seed(cfg.seed)
    print("dataset samples=", len(dataset))
    train_loader = torch.utils.data.DataLoader(dataset, **cfg.loader,shuffle=False, drop_last=False, generator=rnd_gen)
    val_loader = train_loader
    print("number of batches in data loader=", len(train_loader))

    print("\n\nInspect a batch from the dataloader")

    sample = dataset[0]
    print("sample[0][pixels.shape]=", sample["pixels"].shape)

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    batch_sanity_checks = 0
    for batch in train_loader:
        print(batch["pixels"].shape)

        for idx, batch_item_pixels in enumerate(batch["pixels"]):
            for i, frame in enumerate(batch_item_pixels):
                arr = frame.numpy()
                arr = np.transpose(arr, (1,2,0))

                print(arr.shape)
                print(arr.dtype)
                print(arr.min(), arr.max())

                if DEPTH:
                    arr = np.clip(arr, 0, 1)
                    arr = (255 * arr).astype(np.uint8)
                else:
                    arr = arr * std + mean
                    arr = np.clip(arr, 0, 1)
                    arr = (255 * arr).astype(np.uint8)
                im = Image.fromarray(arr)
                im.save(os.path.join(CEM_OUTPUT_DIR, f'loader_batch_{idx}_frame_{i}.png'))
                print("saved to: ", os.path.join(CEM_OUTPUT_DIR, f'loader_batch_{idx}_frame_{i}.png'))
        
            batch_sanity_checks += 1
            if batch_sanity_checks >= 10:    
                break           # break from within this batch

        break       # break from data loader batch


    ##############################
    ##       model / optim      ##
    ##############################

    world_model_JEPA = instantiate_world_model(cfg)

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
        model = world_model_JEPA,
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

    # logger = None
    # if cfg.wandb.enabled:
    #     logger = WandbLogger(**cfg.wandb.config)
    #     logger.log_hyperparams(OmegaConf.to_container(cfg))

    csv_logger = CSVLogger(save_dir="logs", name="csv")

    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.yaml", "w") as f:
        OmegaConf.save(cfg, f)

  
    state_dict = torch.load(CKPT_PATH, map_location="cpu")
    world_model.model.load_state_dict(state_dict, strict=True)
    print(f"\n\nLoaded pretrained weights into the model from {CKPT_PATH}. Evaluating.")     

    #########################################################
    # CEM OVERFIT EXPERIMENT
    #########################################################
    
    from types import SimpleNamespace
    from stable_worldmodel.solver import CEMSolver
    import gymnasium as gym
    import ogbench
    import mujoco

    print("\n\n\nROLLOUT AND CEM EXPERIMENTS\n\n\n")

    world_model_JEPA.eval()
    world_model_JEPA.to("cuda")
    print("world_model_JEPA.type=", type(world_model_JEPA))
    print("world_model_JEPA.device=", next(world_model_JEPA.parameters()).device)

    # -------------------------------------------------------
    # 1. Get one (random) sample batch 
    # ------------------------------------------------------- 

    print("\nGet one over-fit train sample again")
    full_batch = next(iter(train_loader))                                         # we get the (ONE) over-fit training sample again
    print("full_batch.keys()=", full_batch.keys())                              # the sample should contain pixels + actions
    print("full_batch['pixels'].shape= ", full_batch["pixels"].shape)           # and have a shape with num_steps T = history_size + num_preds as: 
    print("full_batch['action'].shape= ", full_batch["action"].shape)           # pixels: [T, 3, 224, 224] and actions: [T, 25]

    with torch.no_grad():
        src_pixels = full_batch["pixels"][0:1].cuda()
        input_actions = full_batch["action"][0:1].cuda()                      # input action sequence has shape [B,T,A]   where T=num_steps, A=action_dim

                                                                    # => 
    # act_sequence_0 = input_actions[:, 0:1].cuda()                   # we want only T=0, and skip T=1                                
    print("src_pixels.shape=", src_pixels.shape)
    print("src_actions.shape =", input_actions.shape)

    encoder_input_dict = {                                   # build the info_dict as expected by the JEPA rollout
        "pixels": src_pixels,
        "action": input_actions,
    }

    
    ''' Run a sanity check before encoding anything '''
    for frame, frame_idx in zip(src_pixels[0], range(src_pixels.shape[1])):

        arr = frame.cpu().numpy()
        arr = np.transpose(arr, (1,2,0))

        if DEPTH:
            arr = np.clip(arr, 0, 1)
            arr = (255 * arr).astype(np.uint8)
        else:
            arr = arr * std + mean
            arr = np.clip(arr, 0, 1)
            arr = (255 * arr).astype(np.uint8)

        im = Image.fromarray(arr)
        im.save(f'{CEM_OUTPUT_DIR}/sanity_check_encoder_input_frame_{frame_idx}.png')


    encoder_output = world_model_JEPA.encode(encoder_input_dict)                           # should return [B,T,D]
    print("\nworld_model_JEPA.encode(encoder_input_batch):")
    print("encoder_output['emb'].shape= ", encoder_output["emb"].shape)
    print("encoder_output['act_emb'].shape= ", encoder_output["act_emb"].shape)
    
    emb = encoder_output["emb"]                             # get pixel and action embeddings
    act_emb = encoder_output["act_emb"]

    print(f"\nSplitting embeddings into context with ctx_len={cfg.wm.history_size} and target with n_preds={cfg.wm.num_preds}")
    ctx_emb = emb[:, :cfg.wm.history_size]                                    # split-build context + target
    ctx_act = act_emb[:, :cfg.wm.history_size]

    tgt_latent = emb[:, cfg.wm.num_preds: ].contiguous()

    print("ctx_emb.shape=", ctx_emb.shape)
    print("ctx_act.shape=", ctx_act.shape)
    print("tgt_latent.shape=", tgt_latent.shape)

    print("\nRunning predictor to get the predicted latent embedding from the context embeddings")
    pred_latent = world_model_JEPA.predict(ctx_emb, ctx_act)
    print("pred_latent.shape=", pred_latent.shape)

    print("\ncompute MSE(pred_latent,tgt_latent)")
    overfit_mse = F.mse_loss(pred_latent, tgt_latent)                              # should be close to target latent !
    print("MSE(pred_latent, tgt_latent)=", overfit_mse)
    cos = F.cosine_similarity(
        pred_latent.flatten(1),
        tgt_latent.flatten(1)
    ).mean()

    print("cosine_similarity(pred_latent, tgt_latent)=", cos)

    gt_norm = torch.norm(pred_latent - tgt_latent)
    print("best_norm =", gt_norm.item())

    # -------------------------------------------------------
    # 2. Configure CEM
    # -------------------------------------------------------
    print("\n\nSetup a CEMsolver")
    solver = CEMSolver(
        model=world_model_JEPA,      # your trained world model
        batch_size=1,
        num_samples=100,
        topk=10,
        n_steps=10,
        device="cuda",
    )
    env = gym.make(
        "visual-cube-single-v0",
        terminate_at_goal=False,
        mode="data_collection",
        width=224,
        height=224,
        pixel_transparent_arm=False,
    )
    solver.configure(
        action_space=env.action_space,
        n_envs=1,
        config=SimpleNamespace(
            horizon=1,      # IMPORTANT
            action_block=5, # frameskip
        ),
    )
    print("before:", solver._action_dim, solver.action_dim)
    solver._action_dim = 5
    print("after:", solver._action_dim, solver.action_dim)

    # -------------------------------------------------------
    # 3. Run CEM
    # -------------------------------------------------------

    print("\n\nRun CEM solver for ONE source frame and ONE target frame, with 1 ACTION block (=5 actions) (frameskip=5) to reach the target")

    CEM_info_dict = {
        "pixels": src_pixels[0:1, 0:1].cuda(),              # source history
        "goal":   src_pixels[0:1, 1:2].cuda(),   # target after one block
    }

    CEM_info_dict["action"] = torch.zeros(
        1,      # batch
        1,      # history length
        25,     # action block
        device="cuda"
    )

    outputs = solver.solve(CEM_info_dict, init_action=None)
    cem_action = outputs["actions"]
    print("solver finished\n\n")
    print("CEM outputs['actions'].shape =", cem_action.shape)
    best_cem_action = cem_action[0, 0].numpy()
    print("best_cem_action.shape=", best_cem_action.shape)

    print("min CEM solver cost=", min(outputs["costs"]))
    print("mean CEM solver cost=", np.mean(outputs["costs"]))

    print("\n\nVisualize the distribution of all CEM costs")

    all_cem_costs = outputs['all_costs']
    print("len outputs['all_costs']", len(all_cem_costs))

    plt.hist(all_cem_costs, bins=50)
    plt.xlabel("MSE Cost")
    plt.axvline(
        overfit_mse.item(),
        linestyle="--",
        color="orange",
        label="A0 loss"
    )
    plt.ylabel("Count")
    plt.title("Distribution of all CEM costs")
    plt.legend()
    plt.savefig(os.path.join(CEM_OUTPUT_DIR, "all_CEM_costs.png"))

    # -------------------------------------------------------
    # 4. Compare to true action
    # -------------------------------------------------------
    
    #### Re-run evaluation step, repeating the loss computation as in training BUT WITH THE ACTION FROM THE CEM SOLVER 
    # Run the encoder for frame0 + CEM_actions=[a0,a1,a2,a3,a4] + frame5 -> returns the latent embedding

    print("\n\nRe-run the encoder with the CEM action sequence to get the predicted latent embedding")

    # Encode and rollout with the best action
    print("\n Rollout with best action: best_action.shape=",cem_action.shape)
    cem_action = cem_action.to(device="cuda", dtype=torch.float32)
    best_act_emb = world_model_JEPA.action_encoder(cem_action)
    print("best_act_emb.shape=", best_act_emb.shape)

    print("\n\n Run predict(ctx_emb, CEM_act_emb) to get the predicted latent from the CEM solver best action")
    world_model_pred_best_latent = world_model_JEPA.predict(ctx_emb,best_act_emb)
    print("world_model_pred_best_emb.shape=", world_model_pred_best_latent.shape)

    best_loss = F.mse_loss(world_model_pred_best_latent, tgt_latent)
    print("MSE(best_latent, tgt_latent) =", best_loss.item())

    cos = F.cosine_similarity(
        world_model_pred_best_latent.flatten(1),
        tgt_latent.flatten(1)
    ).mean()
    print("cosine_similarity(best_latent, tgt_latent)=", cos)

    best_norm = torch.norm(world_model_pred_best_latent - tgt_latent)
    print("best_norm =", best_norm.item())


    # -------------------------------------------------------
    # 5. Rollout in OGBench
    # -------------------------------------------------------

    print("\n\n OGBENCH ROLLOUTS")

    OGB_GT = os.path.join(CEM_OUTPUT_DIR, "ogb_gt")
    OGB_CEM = os.path.join(CEM_OUTPUT_DIR, "ogb_cem")
    os.makedirs(OGB_GT, exist_ok=True)
    os.makedirs(OGB_CEM, exist_ok=True)

    source_qpos = SRC_QPOS_0
    source_qvel = SRC_QVEL_0

    env.reset()
    env.unwrapped.set_state(
        source_qpos,
        source_qvel
    )

    img = env.unwrapped.render(camera="front_pixels", depth=DEPTH)

    if DEPTH: 
        if LINEAR:
            img = np.clip(img, 0.5, 3.0)
            img = (img-0.5)/2.5
        elif INVERSE: 
            img = np.clip(img,0.5,3.0)
            img = 1/img
            img = (img-img.min())/(img.max()-img.min())

        img = (img * 255).astype(np.uint8)
        img = np.repeat(img[..., None], 3, axis=-1)
   
    Image.fromarray(img).save(os.path.join(OGB_GT, f"ogbench_renders_gt_step_0.png"))
    Image.fromarray(img).save(os.path.join(OGB_CEM, f"ogbench_renders_cem_step_0.png"))

    print("input_actions.shape=", input_actions.shape)
    ground_truth_dataset_action = input_actions[0, 0].cpu().numpy()  # the true action sequence from the dataset for this sample
    ground_truth_dataset_action = ground_truth_dataset_action.reshape(5, 5)  # reshape to (action_block, action_dim)

    print("ground_truth_dataset_action,shape=", ground_truth_dataset_action.shape)
    print(ground_truth_dataset_action)
    print("gt actions.shape=", ground_truth_dataset_action.shape)
    for i,a in enumerate(ground_truth_dataset_action):
        print("taking step=", i)
        env.step(a)
        img = env.unwrapped.render(camera="front_pixels", depth=DEPTH)

        if DEPTH: 
            if LINEAR:
                img = np.clip(img, 0.5, 3.0)
                img = (img-0.5)/2.5
            elif INVERSE: 
                img = np.clip(img,0.5,3.0)
                img = 1/img
                img = (img-img.min())/(img.max()-img.min())

            img = (img * 255).astype(np.uint8)
            img = np.repeat(img[..., None], 3, axis=-1)

        Image.fromarray(img).save(os.path.join(OGB_GT, f"ogbench_renders_gt_step_{i+1}.png"))

    cem_action_seq = cem_action.reshape(5, 5).cpu().numpy()
    print("CEM action seq.shape=", cem_action_seq.shape)
    print(cem_action_seq)

    env.reset()
    env.unwrapped.set_state(
        source_qpos,
        source_qvel
    )
    for i,a in enumerate(cem_action_seq):
        print("taking step=", i)
        env.step(a)
        img = env.unwrapped.render(camera="front_pixels", depth=DEPTH)
        if DEPTH: 
            if LINEAR:
                img = np.clip(img, 0.5, 3.0)
                img = (img-0.5)/2.5
            elif INVERSE: 
                img = np.clip(img,0.5,3.0)
                img = 1/img
                img = (img-img.min())/(img.max()-img.min())

            img = (img * 255).astype(np.uint8)
            img = np.repeat(img[..., None], 3, axis=-1)

        Image.fromarray(img).save(os.path.join(OGB_CEM, f"ogbench_renders_cem_step_{i+1}.png")) 

    ####################################################### OGB visualizations 

    visualize_cem_solver.create_vis(CEM_OUTPUT_DIR)

    print("\n\n Outputs in ", CEM_OUTPUT_DIR)

    return

if __name__ == "__main__":
    run()
