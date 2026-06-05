import h5py
import numpy as np
import torch
from tqdm import tqdm

import gymnasium

SOURCE_FILE = "~/data/ogbench/cube_single_expert.h5"
TARGET_FILE = "multiview_data.h5"
EPISODE_IDS_PATH = "/home/student/users/Aaron_workspace/le-wm-3DGeom/models/le-wm/episode_order.pt"
episode_ids = torch.load(EPISODE_IDS_PATH)

NUM_VIEWS = 5
CAMERA_NAMES = ["front", "front_pixels", "left", "right", "top"]

N_EPISODES = 2
episode_ids = episode_ids[:N_EPISODES]

# --------------------------------------------------
# Initialize OGBench environment
# --------------------------------------------------

# Create environment
env = gymnasium.make(
    "visual-cube-single-v0",
    terminate_at_goal=False,
    mode="data_collection",
)
env.reset()

camera_extrinsics = np.zeros((NUM_VIEWS, 4, 4), dtype=np.float32)
camera_intrinsics = np.zeros((NUM_VIEWS, 3, 3), dtype=np.float32)

print(env.unwrapped.model.ncam)
num_cams = 0
for i in range(env.unwrapped.model.ncam):
    print(i, env.unwrapped.model.cam(i).name)
    num_cams += 1 

if num_cams != NUM_VIEWS:
    raise ValueError(f"Expected {NUM_VIEWS} cameras, but found {num_cams}")

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

    for new_ep_id, ep in enumerate(episode_ids):

        ep = int(ep)

        start = ep_offset[ep]
        length = ep_len[ep]

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

            env.set_state(qpos, qvel)

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

print(f"Finished writing {TARGET_FILE}")