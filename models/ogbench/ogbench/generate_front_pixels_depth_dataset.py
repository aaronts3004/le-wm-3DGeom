import h5py
import numpy as np
import torch
from tqdm import tqdm
import ogbench.manipspace
from pathlib import Path
import os
os.environ["MUJOCO_GL"] = "egl"
import matplotlib.pyplot as plt
import mujoco

'''
(FINAL) SCRIPT USED TO GENERATE THE DATASET WITH DEPTH MAPS RENDERED FROM FRONT_PIXELS VIEW
'''

SANITY_CHECKS = 0

from gymnasium.envs.registration import registry

for env_id in sorted(registry.keys()):
    if "cube" in env_id:
        print(env_id)

import gymnasium

def depth_to_normals(depth):
    """
    depth: (H,W) float32 depth in meters
    returns:
        normals: (H,W,3) in [-1,1]
    """
    dzdy, dzdx = np.gradient(depth)
    normals = np.dstack((-dzdx, -dzdy, np.ones_like(depth)))
    normals /= np.linalg.norm(normals, axis=2, keepdims=True) + 1e-8
    return normals

def depth_to_normals_with_intrinsics(depth, K):
    """
    depth : (H,W)
    K     : (3,3)

    returns
        normals : (H,W,3)
    """

    fx = K[0,0]
    fy = K[1,1]
    cx = K[0,2]
    cy = K[1,2]

    H, W = depth.shape

    u, v = np.meshgrid(
        np.arange(W, dtype=np.float32),
        np.arange(H, dtype=np.float32),
        indexing="xy",
    )

    X = (u - cx) * depth / fx
    Y = (v - cy) * depth / fy
    Z = depth

    V = np.stack((X, Y, Z), axis=-1)

    normals = np.zeros_like(V)

    dx = V[:,2:] - V[:,:-2]
    dy = V[2:,:] - V[:-2,:]

    n = np.cross(dx[1:-1], dy[:,1:-1])

    n /= np.linalg.norm(n, axis=-1, keepdims=True) + 1e-8

    normals[1:-1,1:-1] = n

    return normals


def get_camera_intrinsic(model, camera_name, width, height):
    """
    Return camera intrinsic matrix K.
    """

    cam_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_CAMERA,
        camera_name,
    )

    fovy = model.cam_fovy[cam_id]

    fy = height / (2.0 * np.tan(np.deg2rad(fovy) / 2.0))
    fx = fy

    cx = width / 2.0
    cy = height / 2.0

    # K = np.array([
    #     [-fx, 0,  cx],
    #     [0,  fy, cy],
    #     [0,  0,  1 ],
    # ], dtype=np.float32)
    K = np.array([
        [fx, 0,  cx],
        [0,  fy, cy],
        [0,  0,  1 ],
    ], dtype=np.float32)

    return K

from gymnasium.envs.registration import registry

for env_id in sorted(registry.keys()):
    if "cube" in env_id:
        print("Found env_id for CUBE in environment")
        print(env_id)

def sanity_check_normals(normals, normals_u8, angles): 
    print("mean angular error:", angles.mean())

    print("map min and max: ")
    print(normals_u8.min())
    print(normals_u8.max())
    plt.imshow(normals_u8)
    plt.savefig(f"normal_map_filtered_{write_idx}.png")
    plt.close()

    length = np.linalg.norm(normals, axis=-1)

    print(f"min  = {length.min():.5f}")
    print(f"max  = {length.max():.5f}")
    print(f"mean = {length.mean():.5f}")
    print(f"std  = {length.std():.5f}")

    for i, c in enumerate("xyz"):
        print("dimension; ", c)
        print(
            c,
            normals[..., i].min(),
            normals[..., i].max(),
            normals[..., i].mean(),
            normals[..., i].std(),
        )

    table = normals[150:220, 20:200]
    print("table mean: ", table.mean(axis=(0, 1)))

    dot_x = np.sum(normals[:,1:] * normals[:,:-1], axis=-1)
    dot_y = np.sum(normals[1:] * normals[:-1], axis=-1)

    print("neighbour means: ", dot_x.mean(), dot_y.mean())

    grad = np.sqrt(
        np.gradient(depth, axis=0)**2 +
        np.gradient(depth, axis=1)**2
    )

    normal_change = 1 - dot_x
    print("grad: ", grad)
    print("normal change: ", normal_change)

    y = 180
    x = 100
    print("normals xy: ", normals[y, x])

    print("nans: ", np.isnan(normals).sum())
    print("inf: ", np.isinf(normals).sum())

    fig, ax = plt.subplots(1,3, figsize=(12,4))

    titles = ["Nx","Ny","Nz"]

    for i in range(3):
        ax[i].imshow(normals[:,:,i], cmap="coolwarm", vmin=-1, vmax=1)
        ax[i].set_title(titles[i])

    plt.savefig(f"vis_per_normal_{write_idx}")
    plt.close()


def sanity_check_depth(depth): 
    print("depth min and max")
    print(depth.min())
    print(depth.max())
    print(np.unique(depth[:5,:5]))
    plt.imshow(depth, vmin=0.5, vmax=2.5)
    plt.colorbar()
    plt.savefig(f"depth_map_filtered_{write_idx}.png")
    plt.close()


import gymnasium

SOURCE_FILE = Path("~/data/ogbench/cube_single_expert.h5").expanduser()
print("Reading h5 from ", SOURCE_FILE)
print("Found: ", Path(SOURCE_FILE).exists())
EPISODE_IDS_PATH = "/home/student/users/Aaron_workspace/le-wm-3DGeom/models/le-wm/episode_order.pt"
episode_ids = torch.load(EPISODE_IDS_PATH)


N_EPISODES_TRAIN = 1000
N_EPISODES_VAL = 100

DEPTH_ONLY = False
NORMALS = True
RGB_DEPTH = False

train_episode_ids = episode_ids[:N_EPISODES_TRAIN]
val_episode_ids = episode_ids[N_EPISODES_TRAIN : N_EPISODES_TRAIN + N_EPISODES_VAL]

selected_episode_ids = episode_ids[:N_EPISODES_TRAIN + N_EPISODES_VAL]


if NORMALS: 
    TARGET_FILE = Path(f"~/data/ogbench/front_pixels_NORMALS_train_{N_EPISODES_TRAIN}_val_{N_EPISODES_VAL}_episodes.h5").expanduser()               # NORMALS
elif DEPTH_ONLY: 
    TARGET_FILE = Path(f"~/data/ogbench/front_pixels_DEPTH_ONLY_train_{N_EPISODES_TRAIN}_val_{N_EPISODES_VAL}_episodes.h5").expanduser()            # DEPTH ONLY
elif RGB_DEPTH: 
    TARGET_FILE = Path(f"~/data/ogbench/front_pixels_RGB_DEPTH_train_{N_EPISODES_TRAIN}_val_{N_EPISODES_VAL}_episodes.h5").expanduser()         # RGB+DEPTH
else: 
    TARGET_FILE = Path(f"~/data/ogbench/front_pixels_RGB_ONLY_train_{N_EPISODES_TRAIN}_val_{N_EPISODES_VAL}_episodes.h5").expanduser()          # RGB --- ATTENTION: un-tested mode

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


print(env.unwrapped.model.ncam)
num_cams = 0
for i in range(env.unwrapped.model.ncam):
    print(i, env.unwrapped.model.cam(i).name)
    num_cams += 1 

available_cameras = {
    env.unwrapped.model.cam(i).name
    for i in range(env.unwrapped.model.ncam)
}

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

    
    if NORMALS: 
        pixels_ds = f_tgt.create_dataset(
        "pixels",
        shape=(total_frames, H, W, 3),
        dtype=np.float32,
        chunks=(8, H, W, 3),
        compression="lzf",
    )
    elif DEPTH_ONLY:
        pixels_ds = f_tgt.create_dataset(
            "pixels",
            shape=(total_frames, H, W, 1),
            dtype=np.float32,
            chunks=(8, H, W, 1),
            compression="lzf",
        )
    elif RGB_DEPTH: 
        pixels_ds = f_tgt.create_dataset(
        "pixels",
        shape=(total_frames, H, W, 4),
        dtype=np.float32,
        chunks=(8, H, W, 4),
        compression="lzf",
    )
    else: 
        print("Un-tested modality - please implement first (line 231)")
        exit(1)

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
    # Copy data and render depth maps
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
            
            # ------------------------------------------
            # Render RGB / DEPTH / NORMALS / RGB+D
            # ------------------------------------------

            rgb = env.unwrapped.render(camera="front_pixels")
            depth = env.unwrapped.render(camera="front_pixels",depth=True,)
            depth = np.clip(depth, 0.5, 3.0)

            if NORMALS: 
                model = env.unwrapped.model
                K = get_camera_intrinsic(model, "front_pixels", 224, 224)

                n_grad = depth_to_normals(depth)                                    # compare normals with / without intrinsics
                normals   = depth_to_normals_with_intrinsics(depth, K)

                dot = np.sum(n_grad * normals, axis=-1)
                angles = np.degrees(np.arccos(np.clip(dot, -1, 1)))

                normals_u8 = ((normals + 1.0) * 127.5).round().astype(np.uint8)
                pixels_ds[write_idx] = normals_u8

                if write_idx % 5000 == 0: 
                    sanity_check_normals(normals, normals_u8=normals_u8, angles=angles)
                    
            elif DEPTH_ONLY: 
                depth = np.clip(depth, 0.5, 3.0)
                pixels_ds[write_idx] = depth.astype(np.float32)[..., None]

                if write_idx % 5000 == 0: 
                    sanity_check_depth(depth)

            elif RGB_DEPTH:
                rgbd = np.concatenate([rgb, depth[..., None].astype(np.float32)],axis=-1)
                pixels_ds[write_idx] = rgbd
            else:       # RGB
                pixels_ds[write_idx] = rgb
                    
            new_ep_idx.append(ep)
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
        data=np.asarray(selected_episode_ids, dtype=np.int32)
    )

print(f"Finished writing {TARGET_FILE}")
assert write_idx == total_frames
assert len(new_ep_offset) == len(selected_episode_ids)
assert len(new_ep_len) == len(selected_episode_ids)
assert len(new_ep_idx) == total_frames