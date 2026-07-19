import h5py
import numpy as np
import gymnasium
import ogbench.manipspace
from pathlib import Path
from tqdm import tqdm
import os
from PIL import Image
import imageio

os.environ["MUJOCO_GL"] = "egl"

# ==========================================================
# Config
# ==========================================================

SOURCE_FILE = Path("~/data/ogbench/cube_single_expert.h5").expanduser()

FRAME_SKIP = 5
N_SOURCE_STATES = 1
N_ACTION_TRAJECTORIES_PER_STATE = 10
N_ACTION_SEQS = 10
FRAMES_PER_EP = N_ACTION_SEQS + 1 
TOTAL_FRAMES = N_SOURCE_STATES * FRAMES_PER_EP * N_ACTION_TRAJECTORIES_PER_STATE 
ACTION_DIM=5

SEED = 100
rng = np.random.default_rng(SEED)

# ==========================================================
# Environment
# ==========================================================

env = gymnasium.make(
    "visual-cube-single-v0",
    terminate_at_goal=False,
    mode="data_collection",
    width=224,
    height=224,
)

env.reset()


# ==========================================================
# Open source dataset
# ==========================================================

with h5py.File(SOURCE_FILE, "r") as f_src:

    qpos_ds = f_src["qpos"]
    qvel_ds = f_src["qvel"]

    print("total available starting positions: ", len(qpos_ds))
    num_states = len(qpos_ds)
    # ------------------------------------------------------
    # Sample source states
    # ------------------------------------------------------

    source_indices = rng.choice(
        num_states,
        size=N_SOURCE_STATES,
        replace=False,
    )

    TARGET_FILE = Path(
        f"~/data/ogbench/cube_random_{N_SOURCE_STATES}_sources_{N_ACTION_TRAJECTORIES_PER_STATE}_trajectories_{N_ACTION_SEQS}_seqs.h5"
    ).expanduser()
    print("\n\nWriting dataset to:", TARGET_FILE)

    OUTPUT_DIR = Path(f"sanity_checks/overfit_with_actions/cube_{N_SOURCE_STATES}_sources_{N_ACTION_TRAJECTORIES_PER_STATE}_trajectories_{N_ACTION_SEQS}_seqs")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("Output sanity checks to: ", OUTPUT_DIR)


    print("\n\nGenerating dataset:")
    print(f"Number of total source states: {num_states}")
    print(f"Action trajectories: {N_ACTION_TRAJECTORIES_PER_STATE}")
    print(f"Action sequences per trajectory: {N_ACTION_SEQS}")
    print(f"Frames per episode: {FRAMES_PER_EP}")
    

    # ======================================================
    # Create output file
    # ======================================================

    with h5py.File(TARGET_FILE, "w") as f_out:
        f_out.attrs["source_states"] = N_SOURCE_STATES
        f_out.attrs["frame_skip"] = FRAME_SKIP
        f_out.attrs["n_action_trajectories"] = N_ACTION_TRAJECTORIES_PER_STATE
        f_out.attrs["n_seqs_per_source"] = N_ACTION_SEQS

        pixels_ds = f_out.create_dataset(
            "pixels",
            shape=(TOTAL_FRAMES, 224, 224, 3),
            dtype=np.uint8,
            compression="lzf",
        )

        action_ds = f_out.create_dataset(
            "action",
            shape=(TOTAL_FRAMES, FRAME_SKIP * 5),
            dtype=np.float32,
            compression="lzf",
        )

        source_idx_ds = f_out.create_dataset(
            "source_indices",
            shape=(N_SOURCE_STATES,),
            dtype=np.int32,
        )

        qpos_ds_out = f_out.create_dataset(
            "qpos_at_src_idx",
            shape=(N_SOURCE_STATES, 21),
            dtype=np.float32,
        )

        qvel_ds_out = f_out.create_dataset(
            "qvel_at_src_idx",
            shape=(N_SOURCE_STATES, 20),
            dtype=np.float32,
        )

        write_idx = 0

        # ==================================================
        # Main loop
        # ==================================================

        for source_id, src_idx in enumerate(tqdm(source_indices)):              # for SOURCE
            # ----------------------------------------------
            # fixed source state
            # ----------------------------------------------

            qpos0 = qpos_ds[src_idx]
            qvel0 = qvel_ds[src_idx]

            qpos_ds_out[source_id] = qpos0
            qvel_ds_out[source_id] = qvel0
            source_idx_ds[source_id] = src_idx

            env.unwrapped.set_state(qpos0, qvel0)
            src_pixels = env.unwrapped.render(camera="front_pixels")

            print("\nSet initial state to source_idx= ", src_idx)
            im = Image.fromarray(src_pixels)
            im.save(os.path.join(OUTPUT_DIR, f"{src_idx}_src_pixels.png"))

            # ----------------------------------------------
            # sample action chunks
            # ----------------------------------------------

            sanity_checks = 0 
            for traj in range(N_ACTION_TRAJECTORIES_PER_STATE):         # for TRAJECTORY (e.g. 1000)
                
                frames_for_gif = []

                env.unwrapped.set_state(qpos0, qvel0)
                src_pixels = env.unwrapped.render(camera="front_pixels")

                all_pixels = [src_pixels]                                   # [t0, t5, t10, ...]
                all_action_chunks = []
        
                for seq in range(N_ACTION_SEQS):                        # for SEQUENCE (e.g. 2)
                    cur_action_chunk = []
                    act_idx=0
                    for action in range(FRAME_SKIP):                    # for ACTION

                        a = env.action_space.sample()
                        env.step(a)
                        cur_action_chunk.append(a)
                        act_idx += 1

                        if sanity_checks < 3: 
                            act_img = env.unwrapped.render(camera="front_pixels")
                            im = Image.fromarray(act_img)
                            im.save(os.path.join(OUTPUT_DIR, f"{src_idx}_{traj}_action_seq_{seq}_act_{act_idx}.png"))
                            frames_for_gif.append(im)


                    cur_action_chunk = np.asarray(cur_action_chunk, dtype=np.float32)
                    all_action_chunks.append(cur_action_chunk)

                    cur_pixels = env.unwrapped.render(camera="front_pixels")        # render intermediate frame after t=FRAMESKIP actions
                    all_pixels.append(cur_pixels)

                ### sequence is finished 
                
                if sanity_checks < 3: 
                    tgt_pixels = env.unwrapped.render(camera="front_pixels")
                    im = Image.fromarray(tgt_pixels)
                    im.save(os.path.join(OUTPUT_DIR, f"{src_idx}_{traj}_tgt_pixels.png"))
                    sanity_checks += 1 
                    imageio.mimsave(os.path.join(OUTPUT_DIR, f"{src_idx}_{traj}.gif"), frames_for_gif, fps=5)


                    
                # ------------------------------------------
                # save
                # ------------------------------------------

                base = write_idx * FRAMES_PER_EP

                for t in range(FRAMES_PER_EP):
                    idx = base + t
                    pixels_ds[idx] = all_pixels[t]

                    if t < N_ACTION_SEQS:
                        action_ds[idx] = all_action_chunks[t].reshape(-1)
                    else:
                        # final frame has no outgoing action
                        action_ds[idx] = np.zeros(FRAME_SKIP * ACTION_DIM, dtype=np.float32)

 
                write_idx += 1

        

#         expected = N_SOURCE_STATES * N_ACTION_TRAJECTORIES_PER_STATE

#         assert write_idx == expected, (
#             f"Expected {expected} trajectories, "
#             f"generated {write_idx}"
#         )

#         num_eps = expected


#         out_ep_len = np.full(
#             num_eps,
#             FRAMES_PER_EP,
#             dtype=np.int32,
#         )

#         out_ep_offset = np.arange(
#             0,
#             TOTAL_FRAMES,
#             FRAMES_PER_EP,
#             dtype=np.int64,
#         )

#         out_ep_idx = np.repeat(
#             np.arange(num_eps),
#             FRAMES_PER_EP,
#         ).astype(np.int32)

#         f_out.create_dataset("ep_len", data=out_ep_len)
#         f_out.create_dataset("ep_offset", data=out_ep_offset)
#         f_out.create_dataset("ep_idx", data=out_ep_idx)


# print(f"Saved dataset to {TARGET_FILE}")
# with h5py.File(TARGET_FILE, "r") as f_chk:

#     print("\nValidation:")
#     print("pixels:", f_chk["pixels"].shape)
#     print("action:", f_chk["action"].shape)
#     print("qpos:", f_chk["qpos_at_src_idx"].shape)
#     print("qvel:", f_chk["qvel_at_src_idx"].shape)

#     print("ep_len[:5] =", f_chk["ep_len"][:5])
#     print("ep_offset[:5] =", f_chk["ep_offset"][:5])