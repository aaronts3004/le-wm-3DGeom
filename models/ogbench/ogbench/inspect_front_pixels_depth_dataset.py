import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


ORIGINAL_FILE = Path("/home/student/data/ogbench/datasets/ogbench/cube_single_expert.h5").expanduser()

FILE = Path("/home/student/data/ogbench/front_pixels_depth_train_1000_val_100_episodes.h5").expanduser()


with h5py.File(FILE, "r") as f:

    print("========== ATTRIBUTES ==========")
    for k, v in f.attrs.items():
        print(k, ":", v)

    print("\n========== DATASETS ==========")
    for k in f.keys():
        print(f"{k:25s} {f[k].shape} {f[k].dtype}")

with h5py.File(FILE, "r") as f:

    ep_len = f["ep_len"][:]
    ep_offset = f["ep_offset"][:]

    print("Episodes:", len(ep_len))
    print("Total frames:", ep_len.sum())

    print("Min episode length:", ep_len.min())
    print("Max episode length:", ep_len.max())
    print("Mean episode length:", ep_len.mean())

    print("\nFirst 10 episode lengths")
    print(ep_len[:10])

    print("\nFirst 10 offsets")
    print(ep_offset[:10])


with h5py.File(FILE, "r") as f:

    depth = f["pixels"]
    print(depth.shape)
    print(type(depth))

    mins = []
    maxs = []

    for i in range(0, len(depth), 500):
        d = depth[i]

        mins.append(d.min())
        maxs.append(d.max())

    print("Global sampled min:", np.min(mins))
    print("Global sampled max:", np.max(maxs))


import random

indices = []

with h5py.File(FILE, "r") as f:

    depth = f["pixels"]

    fig, axs = plt.subplots(2,2, figsize=(12,8))

    for ax in axs.flat:
        idx = random.randrange(len(depth))
        ax.imshow(depth[idx,...,0], cmap="viridis", vmin=0.5, vmax=3.0)
        ax.set_title(f"Frame {idx}")
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(f"sanity_check_{idx}.png")

    x = depth[0]
    print(x.min(), x.max(), x.mean(), x.std())

import random
import h5py
import matplotlib.pyplot as plt

N = 4

with h5py.File(FILE, "r") as f_depth, h5py.File(ORIGINAL_FILE, "r") as f_rgb:

    depth = f_depth["pixels"]
    rgb = f_rgb["pixels"]
    print("Depth length:", len(depth))
    print("RGB length:", len(rgb))

    print(f_depth.keys())
    print(f_rgb.keys())

    indices = random.sample(range(len(depth)), N)

    fig, axs = plt.subplots(2, N, figsize=(4*N, 8))

    for i, idx in enumerate(indices):

        # Depth
        axs[0, i].imshow(depth[idx, ..., 0], cmap="viridis", vmin=0.5, vmax=3.0)
        axs[0, i].set_title(f"Depth {idx}")
        axs[0, i].axis("off")

        # RGB
        axs[1, i].imshow(rgb[idx])
        axs[1, i].set_title(f"RGB {idx}")
        axs[1, i].axis("off")

    plt.tight_layout()
    plt.savefig("rgb_vs_depth.png", dpi=300)

    print("Depth stats:")
    x = depth[0]
    print(x.min(), x.max(), x.mean(), x.std())

    print("RGB stats:")
    x = rgb[0]
    print(x.min(), x.max(), x.mean(), x.std())


print("depth stats:")
with h5py.File(FILE, "r") as f:
    d = f["pixels"]
    print("dtype:", d.dtype)
    print("shape:", d.shape)
    print("chunks:", d.chunks)
    print("compression:", d.compression)
    print("compression_opts:", d.compression_opts)

print("original stats:")
with h5py.File(ORIGINAL_FILE, "r") as f:
    d = f["pixels"]
    print("dtype:", d.dtype)
    print("shape:", d.shape)
    print("chunks:", d.chunks)
    print("compression:", d.compression)
    print("compression_opts:", d.compression_opts)

import time
import h5py

# print("original time: ")
# with h5py.File(FILE, "r") as f:
#     d = f["pixels"]

#     t0 = time.time()
#     x = d[:1000]
    
#     print(time.time() - t0)


# print("original stats: ")
# with h5py.File(ORIGINAL_FILE, "r") as f:
#     d = f["pixels"]

#     t0 = time.time()
#     x = d[:1000]
#     print(time.time() - t0)

