import h5py
import numpy as np
import torch
from tqdm import tqdm
import ogbench.manipspace
from pathlib import Path
import os
os.environ["MUJOCO_GL"] = "egl"

'''
(FINAL) SCRIPT USED TO GENERATE THE DATASET CURRENTLY IN /data/ogbench/multiview_data_{N_EPISODES_VAL}_val_episodes.h5
'''


from gymnasium.envs.registration import registry

for env_id in sorted(registry.keys()):
    if "cube" in env_id:
        print(env_id)

import gymnasium

SOURCE_FILE = Path("~/data/ogbench/cube_single_expert.h5").expanduser()
print("Reading h5 from ", SOURCE_FILE)
print("Found: ", Path(SOURCE_FILE).exists())
EPISODE_IDS_PATH = "/home/student/users/Aaron_workspace/le-wm-3DGeom/models/le-wm/episode_order.pt"
episode_ids = torch.load(EPISODE_IDS_PATH)

CAMERA_NAMES = ["front_zoomed", "front_pixels", "left", "right", "side"]
NUM_VIEWS = len(CAMERA_NAMES)

N_EPISODES_TRAIN = 1000
N_EPISODES_VAL = 100
train_episode_ids = episode_ids[:N_EPISODES_TRAIN]
val_episode_ids = episode_ids[
    N_EPISODES_TRAIN : N_EPISODES_TRAIN + N_EPISODES_VAL
]

episode_ids = val_episode_ids

TARGET_FILE = Path(f"~/data/ogbench/multiview_data_{N_EPISODES_VAL}_val_episodes.h5").expanduser()
print("Writing new h5 to ", TARGET_FILE)
# --------------------------------------------------
# Initialize OGBench environment
# --------------------------------------------------

# Create environment
env = gymnasium.make(
    "visual-cube-single-v0",
    terminate_at_goal=False,
    mode="data_collection",
    width=224,
    height=224,
)
env.reset()

camera_extrinsics = np.zeros((NUM_VIEWS, 4, 4), dtype=np.float32)           # FIXME Correct parameters
camera_intrinsics = np.zeros((NUM_VIEWS, 3, 3), dtype=np.float32)

print(env.unwrapped.model.ncam)
num_cams = 0
for i in range(env.unwrapped.model.ncam):
    print(i, env.unwrapped.model.cam(i).name)
    num_cams += 1 

available_cameras = {
    env.unwrapped.model.cam(i).name
    for i in range(env.unwrapped.model.ncam)
}
for cam in CAMERA_NAMES:
    if cam not in available_cameras:
        raise ValueError(f"Camera {cam} not found")

# --------------------------------------------------
# Open files
# --------------------------------------------------

with h5py.File(SOURCE_FILE, "r") as f_src, \
     h5py.File(TARGET_FILE, "w") as f_tgt:
    
    f_tgt.attrs["num_views"] = NUM_VIEWS
    f_tgt.attrs["camera_names"] = ",".join(CAMERA_NAMES)
    f_tgt.attrs["source_dataset"] = str(SOURCE_FILE)
    
    # ----------------------------------------------
    # Get relevant episode IDs 
    # ----------------------------------------------
    ep_offset = f_src["ep_offset"]
    ep_len = f_src["ep_len"]

    # ----------------------------------------------
    # Determine dimensions
    # ----------------------------------------------
    _, H, W, C = f_src["pixels"].shape

    # ----------------------------------------------
    # Create multiview datasets
    # ----------------------------------------------

    total_frames = 0

    for ep in episode_ids:
        total_frames += int(ep_len[int(ep)])

    print("Selected episodes:", len(episode_ids))
    print("Total frames:", total_frames)

    pixels_mv = f_tgt.create_dataset(
        "pixels_multiview",
        shape=(total_frames, NUM_VIEWS, H, W, C),
        dtype=np.uint8,
        # FIXME change to chunks=(1, NUM_VIEWS, H, W, C) can accelerate writing and reading for sinalgle frame inference
        chunks=(64, 1, H, W, C), 
        compression="lzf"
    )

    # ----------------------------------------------
    # Camera metadata
    # ----------------------------------------------

    f_tgt.create_dataset(
        "camera_names",
        data=np.array(
            CAMERA_NAMES,
            dtype=h5py.string_dtype()
        )
    )

    # FIXME might have problem with this way of saving camera extrinsics and intrinsics.
    f_tgt.create_dataset(
        "camera_extrinsics",
        data=camera_extrinsics
    )

    f_tgt.create_dataset(
        "camera_intrinsics",
        data=camera_intrinsics
    )

    # ----------------------------------------------
    # Generate multiview data
    # ----------------------------------------------

    keys_to_copy = [
        k for k in f_src.keys()
        if k not in [
            "pixels",
            "ep_idx",
            "ep_offset",
            "ep_len",
        ]
    ]

    output_datasets = {}

    for key in keys_to_copy:
        src_ds = f_src[key]
        output_shape = (total_frames,) + src_ds.shape[1:]
        output_datasets[key] = f_tgt.create_dataset(
            key,
            shape=output_shape,
            dtype=src_ds.dtype,
            compression="lzf"
        )

    qpos_ds = f_src["qpos"]
    qvel_ds = f_src["qvel"]

    new_ep_offset = []
    new_ep_len = []
    new_ep_idx = []

    write_idx = 0

    for new_ep_id, ep in enumerate(tqdm(episode_ids, desc="Episodes")):

        ep = int(ep)

        start = ep_offset[ep]
        length = ep_len[ep]

        # FIXME shouldn't be new_ep_offset.append(start)?
        new_ep_offset.append(write_idx)
        new_ep_len.append(length)
        

        for src_idx in range(start, start + length):
            qpos = qpos_ds[src_idx]
            qvel = qvel_ds[src_idx]

            # -------------------------------------------
            # Copy other datasets 
            # -------------------------------------------
            for key in keys_to_copy:
                output_datasets[key][write_idx] = f_src[key][src_idx]
            # ------------------------------------------
            # Restore simulator state
            # ------------------------------------------

            env.unwrapped.set_state(qpos, qvel)
            # FIXME mujoco.mj_forward(model, data)

            
            # ------------------------------------------
            # Render all views
            # ------------------------------------------

            rgb_views = []
            for camera_name in CAMERA_NAMES:
                img = env.unwrapped.render(camera=camera_name)
                rgb_views.append(img)

            rgb_views = np.stack(rgb_views, axis=0)
            pixels_mv[write_idx] = rgb_views
            new_ep_idx.append(new_ep_id)
            write_idx += 1


    new_ep_offset = np.asarray(new_ep_offset, dtype=np.int64)
    new_ep_len = np.asarray(new_ep_len, dtype=np.int32)
    new_ep_idx = np.asarray(new_ep_idx, dtype=np.int32)

    f_tgt.create_dataset(
        "ep_offset",
        data=new_ep_offset
    )

    f_tgt.create_dataset(
        "ep_len",
        data=new_ep_len
    )

    f_tgt.create_dataset(
        "ep_idx",
        data=new_ep_idx
    )

    f_tgt.create_dataset(
        "original_episode_ids",
        data=np.asarray(episode_ids, dtype=np.int32)
    )

print(f"Finished writing {TARGET_FILE}")