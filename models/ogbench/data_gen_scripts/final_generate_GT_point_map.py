import hdf5plugin
import h5py
import numpy as np
import torch
from tqdm import tqdm
import ogbench.manipspace
from pathlib import Path
import os
os.environ["MUJOCO_GL"] = "egl"

from PIL import Image

from utils import extract_data, _depths_to_world_points_with_colors

'''
(FINAL) SCRIPT USED TO GENERATE THE DATASET WITH:
    1. Ground truth pixel-aligned points map as "pixels" with shape (N, 224, 224, 3), dtype: np.float32
    2. Ground truth color map as "RGB" with shape (N, 224, 224, 3), dtype: np.uint8
    3. During the point reconstruction, pixels with invalid depth will be masked out, 
        thus need attention mask for the model as "atten_mask" with shape (N, 224, 224), dtype: bool
        (notice: for OGBench ground truth, we don't need this mask)
    4. Other datasets copied from original dataset
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


N_EPISODES_TRAIN = 1000
N_EPISODES_VAL = 100
# N_EPISODES_TRAIN = 100
# N_EPISODES_VAL = 10
train_episode_ids = episode_ids[:N_EPISODES_TRAIN]
val_episode_ids = episode_ids[N_EPISODES_TRAIN : N_EPISODES_TRAIN + N_EPISODES_VAL]

selected_episode_ids = episode_ids[:N_EPISODES_TRAIN + N_EPISODES_VAL]

TARGET_FILE = Path(f"~/data/ogbench/gt_point_map_{N_EPISODES_TRAIN}_val_{N_EPISODES_VAL}_episodes.h5").expanduser()
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

CAMERA_NAMES = ["front_pixels"]
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

    gt_points = f_tgt.create_dataset(
        "pixels",
        shape=(total_frames, H, W, 3),
        dtype=np.float32,
        chunks=(1, H, W, 3),
        compression="lzf",
    )

    gt_color = f_tgt.create_dataset(
        "RGB",
        shape=(total_frames, H, W, 3),
        dtype=np.uint8,
        chunks=(1, H, W, 3),
        compression="lzf",
    )

    gt_mask = f_tgt.create_dataset(
        "atten_mask",
        shape=(total_frames, H, W),
        dtype=bool,
        chunks=(1, H, W),
        compression="lzf",
    )

    keys_to_copy = [
        k for k in f_src.keys()
        if k not in ["pixels", "ep_idx", "ep_offset", "ep_len"]
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

        ep_pts3d = []
        ep_color = []
        ep_atten_mask = []

        for local_idx in tqdm(range(length), desc=f"Epoch {ep} Processing", leave=False):
            qpos = ep_qpos[local_idx]
            qvel = ep_qvel[local_idx]

            # Restore simulator state
            env.unwrapped.set_state(qpos, qvel)
            
            # Render multiviewrgb and get camera intrinsics / extrinsics
            rgb_views = []
            camera_intrinsics = []
            camera_extrinsics = []
            depth_views = []

            for camera_name in CAMERA_NAMES:
                rgb, K, T, depth = extract_data(env, camera_name)
                rgb_views.append(rgb)
                camera_intrinsics.append(K)
                camera_extrinsics.append(T)

                # print(depth.min(), depth.max())
                depth = np.clip(depth, 0.5, 2.0)
                depth_views.append(depth)

                # print("rgb shape", rgb.shape)
                # print("rgb type", rgb.dtype)
                # print("K shape", K.shape)
                # print("K type", K.dtype)
                # print("T shape", T.shape)
                # print("T type", T.dtype)
                # print("depth shape", depth.shape)
                # print("depth type", depth.dtype)

                '''gb shape (224, 224, 3)   0%|                                                                                                                                                      | 0/201 [00:00<?, ?it/s]
                    rgb type uint8
                    K shape (3, 3)
                    K type float32
                    T shape (4, 4)
                    T type float32
                    depth shape (224, 224)
                    depth type float32
                    '''
            points, colors, attention_mask = _depths_to_world_points_with_colors(
                depth=np.stack(depth_views, axis=0),
                K=np.stack(camera_intrinsics, axis=0),
                ext_w2c=np.stack(camera_extrinsics, axis=0),  # w2c
                images_u8=np.stack(rgb_views, axis=0), 
                conf=None,
                pose="GLB",
            )

            # print("points shape", points.shape)
            # print("points type", points.dtype)
            # print("colors shape", colors.shape)
            # print("colors type", colors.dtype)
            # print("attention_mask shape", attention_mask.shape)
            # print("attention_mask type", attention_mask.dtype)

            '''points shape (224, 224, 3)
                points type float32
                colors shape (224, 224, 3)
                colors type uint8
                attention_mask shape (224, 224)
                attention_mask type bool
                '''
            # print("The attention are all True:", attention_mask.all().item() )
            
            ep_pts3d.append(points)
            ep_color.append(colors)
            ep_atten_mask.append(attention_mask)

        tgt_slice = slice(write_idx, write_idx + length)
            
        gt_points[tgt_slice] = np.stack(ep_pts3d, axis=0)
        gt_color[tgt_slice] = np.stack(ep_color, axis=0)
        gt_mask[tgt_slice] = np.stack(ep_atten_mask, axis=0)

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
