import contextlib
import os
from pathlib import Path

import hdf5plugin  
import h5py
import numpy as np
import torch
from tqdm import tqdm

from depth_anything_3.api import DepthAnything3


'''
(FINAL) SCRIPT USED TO GENERATE THE DATASET WITH:
    1. DA3 inferenced depth based on Triple view (front_pixels, left, right) and camera intrinsics/extrinsics
        saved as "pixels" (total_frames, 224, 224, 1) 
        in the "DA3_triple_depth_{N_EPISODES_TRAIN}_val_{N_EPISODES_VAL}_episodes.h5" file
    2. DA3 inferenced depth based on Single view (front_pixels) and camera intrinsics/extrinsics
        saved as "pixels" (total_frames, 224, 224, 1) 
        in the "DA3_single_depth_{N_EPISODES_TRAIN}_val_{N_EPISODES_VAL}_episodes.h5" file
    3. Other datasets copied from original dataset "triple_views_with_cam_train_{N_EPISODES_TRAIN}_val_{N_EPISODES_VAL}_episodes.h5"
    4. "ep_idx" stores the episode index of the original lewm dataset for each frame,
        a list of accquired episodes is stored as "original_episode_ids"
'''

# N_EPISODES_TRAIN = 1000
# N_EPISODES_VAL = 100
N_EPISODES_TRAIN = 1
N_EPISODES_VAL = 0

SOURCE_FILE = Path(f"~/data/ogbench/triple_views_with_cam_train_{N_EPISODES_TRAIN}_val_{N_EPISODES_VAL}_episodes.h5").expanduser()
print("Reading h5 from ", SOURCE_FILE)
print("Found: ", Path(SOURCE_FILE).exists())

TARGET_FILE_SIN = Path(f"~/data/ogbench/DA3_single_depth_{N_EPISODES_TRAIN}_val_{N_EPISODES_VAL}_episodes.h5").expanduser()
print("Writing new h5 to ", TARGET_FILE_SIN)

TARGET_FILE_TRI = Path(f"~/data/ogbench/DA3_triple_depth_{N_EPISODES_TRAIN}_val_{N_EPISODES_VAL}_episodes.h5").expanduser()
print("Writing new h5 to ", TARGET_FILE_TRI)

# DA_MODEL="DA3Nested-Giant-Large"
DA_MODEL="DA3-Large"
#--------------------------------------------------
# Initialize Depth Anything 3 model
#--------------------------------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"
model = DepthAnything3.from_pretrained(f"depth-anything/{DA_MODEL}").to(device)

# --------------------------------------------------
# Open files
# --------------------------------------------------

with h5py.File(SOURCE_FILE, "r") as f_src, \
     h5py.File(TARGET_FILE_SIN, "w") as f_tgt_sin, \
     h5py.File(TARGET_FILE_TRI, "w") as f_tgt_tri:

    f_tgt_sin.attrs["source_dataset"] = str(SOURCE_FILE)
    f_tgt_tri.attrs["source_dataset"] = str(SOURCE_FILE)

    # ----------------------------------------------
    # Get relevant episode IDs 
    # ----------------------------------------------
    ep_offset = f_src["ep_offset"]
    ep_len = f_src["ep_len"]

    print("ep_offset", ep_offset)

    # ----------------------------------------------
    # Determine dimensions
    # ----------------------------------------------
    num_eps = f_src["original_episode_ids"].shape[0]

    total_frames, H, W, _ = f_src["pixels"].shape

    print("Total episodes:", num_eps)
    print("Total frames:", total_frames)

    depth_ds_sin = f_tgt_sin.create_dataset(
        "pixels",
        shape=(total_frames, H, W, 1),
        dtype=np.float32,
        chunks=(1, H, W, 1),
        compression="lzf",
    )

    depth_ds_tri = f_tgt_tri.create_dataset(
        "pixels",
        shape=(total_frames, H, W, 1),
        dtype=np.float32,
        chunks=(1, H, W, 1),
        compression="lzf",
    )

    keys_to_copy_frame_aligened = [
        k for k in f_src.keys()
        if k not in ["pixels", "triple_pixels", "triple_cam_in", "triple_cam_ex", "ep_len", "ep_offset", "original_episode_ids"]
    ]

    keys_to_copy_ep_aligened = ["ep_len", "ep_offset", "original_episode_ids"]

    output_datasets_sin = {}
    output_datasets_tri = {}

    for key in keys_to_copy_frame_aligened + keys_to_copy_ep_aligened:
        src_ds = f_src[key]
        output_datasets_sin[key] = f_tgt_sin.create_dataset(
            key,
            shape=src_ds.shape,
            dtype=src_ds.dtype,
            compression="lzf"
        )
        output_datasets_tri[key] = f_tgt_tri.create_dataset(
            key,
            shape=src_ds.shape,
            dtype=src_ds.dtype,
            compression="lzf"
        )

    # ----------------------------------------------
    # Copy data and inference depth
    # ----------------------------------------------

    for ep in tqdm(range(num_eps), desc="Episodes"):

        start = ep_offset[ep]
        length = ep_len[ep]
        end = start + length

        ep_images = f_src["pixels"][start:end]
        ep_triple_pixels = f_src["triple_pixels"][start:end]
        ep_triple_cam_in = f_src["triple_cam_in"][start:end]
        ep_triple_cam_ex = f_src["triple_cam_ex"][start:end]

        # -------------------------------------------
        # Copy(cached) other datasets 
        # -------------------------------------------
        cached_src_data = {}
        for key in keys_to_copy_frame_aligened:
            cached_src_data[key] = f_src[key][start:end]

        ep_single_depths = []
        ep_triple_depths = []
        for local_idx in tqdm(range(length), desc=f"Epoch {ep} Processing", leave=False):
            with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
                # ===================================
                # DA3 single view depth estimation
                # ===================================
                single_image = ep_images[local_idx] # (224, 224, 3)
                img_list = [single_image] # list of images for DA3 inference

                # print(f"Processing frame {local_idx} of episode {ep}, image shape: {single_image.shape}")
                prediction_single = model.inference(
                    image=img_list,
                    process_res=H,
                    ref_view_strategy = "saddle_balanced",  # "first", "middle", "saddle_balanced", "saddle_sim_range"
                )

                pred_single_depth = prediction_single.depth[0].astype(np.float32)[..., None]
                # print(f"Predicted single depth shape: {pred_single_depth.shape}, min: {pred_single_depth.min()}, max: {pred_single_depth.max()}")
                ep_single_depths.append(pred_single_depth)

                # ===================================
                # DA3 triple view depth estimation
                # ===================================
                triple_pixels = ep_triple_pixels[local_idx] # (3, 224, 224, 3)
                triple_cam_in = ep_triple_cam_in[local_idx] # (3, 3, 3)
                triple_cam_ex = ep_triple_cam_ex[local_idx] # (3, 4, 4)
                # print(f"Triple pixels shape: {triple_pixels.shape}, Triple cam_in shape: {triple_cam_in.shape}, Triple cam_ex shape: {triple_cam_ex.shape}")

                triple_list = list(triple_pixels) # list of 3 images for DA3 inference
                prediction_triple = model.inference(
                    image=triple_list,
                    extrinsics=triple_cam_ex,             # # for single view per frame inference,we can't provide extr and intr by default,
                    intrinsics=triple_cam_in,
                    align_to_input_ext_scale = True, # i.e. we cannot compute the scale, return only relative depth or not aligned metric depth (can be close to the metric depth if we use DA3-Nested)
                    process_res=H,
                    ref_view_strategy = "saddle_balanced",  # "first", "middle", "saddle_balanced", "saddle_sim_range"
                )

                pred_triple_depth = prediction_triple.depth[0].astype(np.float32)[..., None]
                # print(f"Predicted triple depth shape: {pred_triple_depth.shape}, min: {pred_triple_depth.min()}, max: {pred_triple_depth.max()}")
                ep_triple_depths.append(pred_triple_depth)

        tgt_slice = slice(start, end)

        depth_ds_sin[tgt_slice] = np.stack(ep_single_depths, axis=0)
        depth_ds_tri[tgt_slice] = np.stack(ep_triple_depths, axis=0)

        for key in keys_to_copy_frame_aligened:
            output_datasets_sin[key][tgt_slice] = cached_src_data[key]
            output_datasets_tri[key][tgt_slice] = cached_src_data[key]


    for key in keys_to_copy_ep_aligened:
        print(f"{key} is:", f_src[key][...])
        output_datasets_sin[key][...] = f_src[key][...]
        output_datasets_tri[key][...] = f_src[key][...]
        print(f"{key} is:", f_tgt_sin[key][...])


    for key in f_tgt_sin.keys():
            ds = f_tgt_sin[key]
            print(f"{key} | {str(ds.shape)} | {str(ds.dtype)} | {str(ds.compression)}")

    for key in f_tgt_tri.keys():
            ds = f_tgt_tri[key]
            print(f"{key} | {str(ds.shape)} | {str(ds.dtype)} | {str(ds.compression)}")

print(f"Finished writing {TARGET_FILE_SIN}")
print(f"Finished writing {TARGET_FILE_TRI}")