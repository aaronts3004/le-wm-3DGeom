import h5py
import numpy as np
import matplotlib.pyplot as plt

import h5py
import numpy as np
import matplotlib.pyplot as plt

ORIGINAL_DS = "/home/student/data/ogbench/cube_single_expert.h5"
SRC_DA3 = "/home/student/data/ogbench/DA3_triple_depth_10_val_1_episodes.h5"
SRC_PM = "/home/student/data/ogbench/gt_point_map_10_val_1_episodes.h5"

DEPTH = "/home/student/data/ogbench/front_pixels_depth_train_100_val_10_episodes.h5"

CHECK_CURRENT = SRC_PM
IS_POINT_MAP = True

# --------------------------------------------------
# 1. Dataset structure
# --------------------------------------------------

def inspect_structure(DATASET):

    with h5py.File(DATASET, "r") as f:

        print("\n=== DATASET STRUCTURE ===")

        for k in f.keys():
            print(f"{k}: {f[k].shape}")


        print("\nep_idx[:100]=")
        print(f["ep_idx"][:100])

        # print("\noriginal_episode_ids:")
        # print(f["original_episode_ids"][:100])

        print("\nep_offset:")
        print(f["ep_offset"][:100])

        print("\nep_len:")
        print(f["ep_len"][:100])

        print("***")
        print("\nep_idx[200:400]=")
        print(f["ep_idx"][200:400])

        # print("\noriginal_episode_ids[200:400]=")
        # print(f["original_episode_ids"][200:400])

        print("\nep_offset[200:400]=")
        print(f["ep_offset"][200:400])

        print("\nep_len[200:400]=")
        print(f["ep_len"][200:400])


        

# --------------------------------------------------
# 2. Episode metadata validation
# --------------------------------------------------

def validate_episode_metadata(DATASET):

    with h5py.File(DATASET, "r") as f:

        print("\n=== EPISODE METADATA ===")

        ep_offset = f["ep_offset"][:]
        ep_len = f["ep_len"][:]
        ep_idx = f["ep_idx"][:]

        for ep in range(len(ep_offset)):

            start = ep_offset[ep]
            length = ep_len[ep]

            unique = set(
                ep_idx[start:start+length]
            )

            print(
                f"episode={ep} "
                f"start={start} "
                f"len={length} "
                f"ids={unique}"
            )


# --------------------------------------------------
# 3. Camera diversity check
# --------------------------------------------------


# --------------------------------------------------
# 4. Original dataset consistency
# --------------------------------------------------

def validate_original_mapping(DATASET_SRC, DATASET_NEW):

    with h5py.File(DATASET_SRC, "r") as src, \
         h5py.File(DATASET_NEW, "r") as mv:

        print("\n=== ORIGINAL EPISODE MAPPING ===")

        orig_ep = int(
            mv["original_episode_ids"][0]
        )

        src_start = int(
            src["ep_offset"][orig_ep]
        )

        qpos_close = np.allclose(
            src["qpos"][src_start],
            mv["qpos"][0]
        )

        qvel_close = np.allclose(
            src["qvel"][src_start],
            mv["qvel"][0]
        )

        print("original episode:", orig_ep)
        print("qpos close:", qpos_close)
        print("qvel close:", qvel_close)

        if not qpos_close:

            diff = np.abs(
                src["qpos"][src_start]
                - mv["qpos"][0]
            )

            print(
                "max qpos diff:",
                diff.max()
            )

            print(
                "mean qpos diff:",
                diff.mean()
            )


# --------------------------------------------------
# 5. Save visual sanity image
# --------------------------------------------------

def save_visualization(DATASET):
    frame_num = 105
    with h5py.File(DATASET, "r") as f:

        img = f["pixels"][frame_num]
        plt.tight_layout()
        plt.savefig(
            f"DA3_sanity_episode_id_{frame_num}_VAL.png",
            dpi=150
        )

        print("\nSaved multiview_sanity.png")

# --------------------------------------------------
# 5. Save depth sanity image
# --------------------------------------------------
def save_depth_with_colorbar(DATASET):
    frame_num = 105
    with h5py.File(DATASET, "r") as f:
        depth_map = f["pixels"][frame_num]

        fig, ax = plt.subplots(figsize=(6, 5))  

        im = ax.imshow(
            depth_map,
            cmap="inferno",
            aspect="equal",
            vmin=depth_map.min(),
            vmax=depth_map.max(),
        )

        ax.axis("off")

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.set_ylabel("Depth (meters)", rotation=-90, va="bottom")

        output_name = f"depth_single_with_scale_frame_{frame_num}.png"
        plt.savefig(output_name, dpi=150, bbox_inches="tight")

        plt.close(fig)
        print(f"Saved with colorbar: {output_name}")


        print("depth.shape=", depth_map.shape)
        print("depth.min=", depth_map.min())
        print("depth.max=", depth_map.max())
        print("depth.mean=", depth_map.mean())


def get_global_statistics(DATASET):
    
    print("Get global dataset statistics")
    with h5py.File(DATASET, "r") as f:

        if IS_POINT_MAP:
            pts = f["pixels"]   # (N, H, W) or (N, H, W, 1)

            x = pts[..., 0]
            y = pts[..., 1]
            z = pts[..., 2]

            for i,name in enumerate(["X","Y","Z"]):
                arr = pts[...,i]

                print(name)
                print(arr.min())
                print(arr.max())
                print(arr.mean())
                print(arr.std())
                print(np.percentile(arr,1))
                print(np.percentile(arr,99))

            pts = pts[0]
            fig, ax = plt.subplots(1,3, figsize=(15,5))

            titles = ["X", "Y", "Z"]

            for i in range(3):
                im = ax[i].imshow(pts[...,i], cmap="viridis")
                ax[i].set_title(titles[i])
                fig.colorbar(im, ax=ax[i])

            plt.savefig("point_map_vis.png")


        else: 
            depths = f["pixels"]

            n_pixels = 0
            sum_depth = 0.0
            sum_sq_depth = 0.0

            global_min = np.inf
            global_max = -np.inf

            sample_values = []

            print("ds['pixels'].shape=", depths.shape)

            for i in range(len(depths)):
                d = depths[i].astype(np.float64)

                global_min = min(global_min, d.min())
                global_max = max(global_max, d.max())

                sum_depth += d.sum()
                sum_sq_depth += (d ** 2).sum()
                n_pixels += d.size

            mean = sum_depth / n_pixels
            var = sum_sq_depth / n_pixels - mean**2
            std = np.sqrt(var)

            print(f"min  = {global_min:.4f}")
            print(f"max  = {global_max:.4f}")
            print(f"mean = {mean:.4f}")
            print(f"std  = {std:.4f}")

            print("1% :", np.percentile(depths, 1))
            print("99%:", np.percentile(depths, 99))


# --------------------------------------------------
# Run all checks
# --------------------------------------------------



inspect_structure(CHECK_CURRENT)
validate_episode_metadata(CHECK_CURRENT)
# validate_camera_diversity()
validate_original_mapping(ORIGINAL_DS, CHECK_CURRENT)
# save_visualization()
save_depth_with_colorbar(CHECK_CURRENT)
get_global_statistics(CHECK_CURRENT)