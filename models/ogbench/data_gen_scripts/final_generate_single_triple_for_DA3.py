import hdf5plugin
import h5py
import numpy as np
import torch
from tqdm import tqdm
import ogbench.manipspace
from pathlib import Path
import os
os.environ["MUJOCO_GL"] = "egl"

from utils import extract_data

'''
(FINAL) SCRIPT USED TO GENERATE THE DATASET WITH:
    1. Triple view (front_pixels, left, right) as "triple_pixels", in which "front_pixels" is the original view
    2. Camera intrinsics for triple view as "triple_cam_in"
    3. Camera extrinsics for triple view as "triple_cam_ex"
    4. Other datasets copied from original dataset, including original single view as "pixels"
'''


# from gymnasium.envs.registration import registry

# for env_id in sorted(registry.keys()):
#     if "cube" in env_id:
#         print("Found env_id for CUBE in environment")
#         print(env_id)

import gymnasium

SOURCE_FILE = Path("~/data/ogbench/cube_single_expert.h5").expanduser()
print("Reading h5 from ", SOURCE_FILE)
print("Found: ", Path(SOURCE_FILE).exists())
EPISODE_IDS_PATH = "/home/student/users/Aaron_workspace/le-wm-3DGeom/models/le-wm/episode_order.pt"
episode_ids = torch.load(EPISODE_IDS_PATH)


# N_EPISODES_TRAIN = 1000
# N_EPISODES_VAL = 100
N_EPISODES_TRAIN = 10
N_EPISODES_VAL = 1
train_episode_ids = episode_ids[:N_EPISODES_TRAIN]
val_episode_ids = episode_ids[N_EPISODES_TRAIN : N_EPISODES_TRAIN + N_EPISODES_VAL]

selected_episode_ids = episode_ids[:N_EPISODES_TRAIN + N_EPISODES_VAL]

TARGET_FILE = Path(f"~/data/ogbench/triple_views_with_cam_train_{N_EPISODES_TRAIN}_val_{N_EPISODES_VAL}_episodes.h5").expanduser()
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
    pixel_transparent_arm=False
)
env.reset()

model = env.unwrapped.model
data = env.unwrapped.data

print(model.ncam)
num_cams = 0
for i in range(model.ncam):
    print(i, model.cam(i).name)
    num_cams += 1 

available_cameras = {
    model.cam(i).name
    for i in range(model.ncam)
}

print("Available cameras in the environment: ", available_cameras)

CAMERA_NAMES = ["front_pixels", "left", "right"]
# --------------------------------------------------
# Open files
# --------------------------------------------------

with h5py.File(SOURCE_FILE, "r") as f_src, \
     h5py.File(TARGET_FILE, "w") as f_tgt:
    
    f_tgt.attrs["source_dataset"] = str(SOURCE_FILE)
    
    # ----------------------------------------------
    # Get relevant episode IDs 
    # ----------------------------------------------
    ep_offset = f_src["ep_offset"]
    ep_len = f_src["ep_len"]

    # ----------------------------------------------
    # Determine dimensions
    # ----------------------------------------------
    _, H, W, _ = f_src["pixels"].shape

    total_frames = 0

    for ep in selected_episode_ids:
        total_frames += int(ep_len[int(ep)])

    print("Selected episodes:", len(selected_episode_ids))
    print("Total frames:", total_frames)

    triple_pixels = f_tgt.create_dataset(
        "triple_pixels",
        shape=(total_frames, 3, H, W, 3),
        dtype=np.uint8,
        chunks=(1, 3, H, W, 3),
        compression="lzf",
    )

    triple_cam_in = f_tgt.create_dataset(
        "triple_cam_in",
        shape=(total_frames, 3, 3, 3),
        dtype=np.float32,
        chunks=(1, 3, 3, 3),
        compression="lzf",
    )

    triple_cam_ex = f_tgt.create_dataset(
        "triple_cam_ex",
        shape=(total_frames, 3, 4, 4),
        dtype=np.float32,
        chunks=(1, 3, 4, 4),
        compression="lzf",
    )

    keys_to_copy = [
        k for k in f_src.keys()
        if k not in ["ep_idx", "ep_offset", "ep_len"]
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

    # ----------------------------------------------
    # Copy data and render multiview data
    # ----------------------------------------------
    qpos_ds = f_src["qpos"]
    qvel_ds = f_src["qvel"]

    new_ep_offset = []
    new_ep_len = []
    new_ep_idx = []

    write_idx = 0

    for ep in tqdm(selected_episode_ids, desc="Episodes"):

        ep = int(ep)

        start = ep_offset[ep]
        length = ep_len[ep]
        end = start + length

        new_ep_offset.append(write_idx)
        new_ep_len.append(length)

        ep_qpos = qpos_ds[start:end]
        ep_qvel = qvel_ds[start:end]

        # -------------------------------------------
        # Copy(cached) other datasets 
        # -------------------------------------------
        cached_src_data = {}
        for key in keys_to_copy:
            cached_src_data[key] = f_src[key][start:end]

        ep_rgb_views = []
        ep_cam_in = []
        ep_cam_ex = []

        for local_idx in tqdm(range(length), desc=f"Epoch {ep} Processing", leave=False):
            qpos = ep_qpos[local_idx]
            qvel = ep_qvel[local_idx]

            # Restore simulator state
            env.unwrapped.set_state(qpos, qvel)
            
            # Render multiviewrgb and get camera intrinsics / extrinsics
            rgb_views = []
            camera_intrinsics = []
            camera_extrinsics = []

            for camera_name in CAMERA_NAMES:
                rgb, K, T, _ = extract_data(env, camera_name)
                rgb_views.append(rgb)
                camera_intrinsics.append(K)
                camera_extrinsics.append(T)
                
            ep_rgb_views.append(np.stack(rgb_views, axis=0))
            ep_cam_in.append(np.stack(camera_intrinsics, axis=0))
            ep_cam_ex.append(np.stack(camera_extrinsics, axis=0))

        tgt_slice = slice(write_idx, write_idx + length)
            
        triple_pixels[tgt_slice] = np.stack(ep_rgb_views, axis=0)
        triple_cam_in[tgt_slice] = np.stack(ep_cam_in, axis=0)
        triple_cam_ex[tgt_slice] = np.stack(ep_cam_ex, axis=0)

        for key in keys_to_copy:
            output_datasets[key][tgt_slice] = cached_src_data[key]

        new_ep_idx.extend([ep] * length)
        write_idx += length


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
        data=np.asarray(selected_episode_ids, dtype=np.int32)
    )

    for key in f_tgt.keys():
            ds = f_tgt[key]
            print(f"{key} | {str(ds.shape)} | {str(ds.dtype)} | {str(ds.compression)}")

print(f"Finished writing {TARGET_FILE}")
assert write_idx == total_frames
assert len(new_ep_offset) == len(selected_episode_ids)
assert len(new_ep_len) == len(selected_episode_ids)
assert len(new_ep_idx) == total_frames
