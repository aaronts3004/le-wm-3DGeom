import h5py
import numpy as np
import gymnasium
import ogbench.manipspace
from pathlib import Path
from tqdm import tqdm
import os
from PIL import Image

os.environ["MUJOCO_GL"] = "egl"

# ==========================================================
# Config
# ==========================================================

SOURCE_FILE = Path("~/data/ogbench/cube_single_expert.h5").expanduser()

FRAME_SKIP = 5
N_SOURCE_STATES = 1
N_ACTION_TRAJECTORIES_PER_STATE  = 1000
N_ACTION_SEQS = 2

FRAMES_PER_CLIP = N_ACTION_SEQS + 1 
TOTAL_FRAMES = N_SOURCE_STATES * FRAMES_PER_CLIP * N_ACTION_TRAJECTORIES_PER_STATE 
TOTAL_SAMPLES = (N_SOURCE_STATES * N_ACTION_TRAJECTORIES_PER_STATE * N_ACTION_SEQS)       # total number of actions 

SEED = 64
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

    actions = f_src["action"]
    qpos_ds = f_src["qpos"]
    qvel_ds = f_src["qvel"]

    num_frames = len(actions)
    print("number of frames in src dataset=", num_frames)
    
    # ------------------------------------------------------
    # Valid action chunk start indices
    # ------------------------------------------------------

    ep_offset = f_src["ep_offset"][:]
    ep_len = f_src["ep_len"][:]
    valid_chunk_starts = []

    for ep in range(len(ep_offset)):

        start = ep_offset[ep]
        end = start + ep_len[ep]

        for idx in range(start, end - FRAME_SKIP):
            valid_chunk_starts.append(idx)

    valid_chunk_starts = np.asarray(valid_chunk_starts)
    print("\n\n! Valid chunk starts=", len(valid_chunk_starts))

    # ------------------------------------------------------
    # Sample source states
    # ------------------------------------------------------

    source_indices = rng.choice(valid_chunk_starts, size=N_SOURCE_STATES,replace=False)
    source_idx = int(source_indices[0])

    TARGET_FILE = Path(
        f"~/data/ogbench/cube_expert_source_idx_{source_idx}_actions_{N_ACTION_TRAJECTORIES_PER_STATE}_{N_ACTION_SEQS}_SEQ.h5"
    ).expanduser()
    print("\n\nWriting dataset to:", TARGET_FILE)

    OUTPUT_DIR = Path(f"sanity_checks/overfit_with_actions/cube_RANDOM_source_idx_{source_idx}_actions_{N_ACTION_TRAJECTORIES_PER_STATE}_{N_ACTION_SEQS}_SEQ")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("Output sanity checks to: ", OUTPUT_DIR)

    print("\n\nGenerating dataset:")
    print("Source states:", N_SOURCE_STATES)
    print("Actions/state:", N_ACTION_TRAJECTORIES_PER_STATE)
    print("Total samples:", TOTAL_SAMPLES)

    # ======================================================
    # Create output file
    # ======================================================

    sanity_checks = 0 

    with h5py.File(TARGET_FILE, "w") as f_out:
        f_out.attrs["source_idx"] = source_idx
        f_out.attrs["frame_skip"] = FRAME_SKIP
        f_out.attrs["n_action_chunks"] = N_ACTION_TRAJECTORIES_PER_STATE
        f_out.attrs["n_seq_per_source"] = N_ACTION_SEQS

        pixel_shape = (224, 224, 3)

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

        for source_id, src_idx in enumerate(tqdm(source_indices)):              # for now only one

            # ----------------------------------------------
            # fixed source state
            # ----------------------------------------------

            qpos0 = qpos_ds[src_idx]
            qvel0 = qvel_ds[src_idx]

            env.unwrapped.set_state(qpos0, qvel0)

            src_pixels = env.unwrapped.render(camera="front_pixels")

            print("\nSet initial state to source_idx= ", src_idx)
            im = Image.fromarray(src_pixels)
            im.save(os.path.join(OUTPUT_DIR, f"{src_idx}_src_pixels.png"))

            # ----------------------------------------------
            # sample many action chunks
            # ----------------------------------------------

            chunk_starts = rng.choice(
                valid_chunk_starts,
                size=N_ACTION_TRAJECTORIES_PER_STATE,
                replace=True,
            )

            for chunk_idx, chunk_start in enumerate(chunk_starts):

                action_chunk = actions[chunk_start:chunk_start + FRAME_SKIP]

                # ------------------------------------------
                # restore SAME source state
                # ------------------------------------------

                env.unwrapped.set_state(qpos0, qvel0,)

                # ------------------------------------------
                # execute action chunk
                # ------------------------------------------
                act_idx = 0
                for a in action_chunk:
                    env.step(a)

                    if sanity_checks < 10: 
                        act_img = env.unwrapped.render(camera="front_pixels")
                        im = Image.fromarray(act_img)
                        im.save(os.path.join(OUTPUT_DIR, f"src_pixels_{src_idx}_action_chunk_{chunk_idx}_act_{act_idx}.png"))
                        act_idx += 1


                

                tgt_pixels = env.unwrapped.render(
                    camera="front_pixels"
                )

                if sanity_checks < 10: 
                    im = Image.fromarray(tgt_pixels)
                    im.save(os.path.join(OUTPUT_DIR, f"tgt_pixels_{src_idx}_chunk_{chunk_idx}.png"))


                # ------------------------------------------
                # save
                # ------------------------------------------

                # frame0_idx = 2 * write_idx
                # frame1_idx = 2 * write_idx + 1

                # flat_action = action_chunk.reshape(-1)

                # # frame 0 = source
                # pixels_ds[frame0_idx] = src_pixels
                # action_ds[frame0_idx] = flat_action
                # qpos_ds_out[frame0_idx] = qpos0
                # qvel_ds_out[frame0_idx] = qvel0

                # # frame 1 = target
                # pixels_ds[frame1_idx] = tgt_pixels
                # action_ds[frame1_idx] = flat_action
                # qpos_ds_out[frame1_idx] = env.unwrapped.data.qpos.copy()
                # qvel_ds_out[frame1_idx] = env.unwrapped.data.qvel.copy()


                write_idx += 1
                sanity_checks += 1

        

#         assert write_idx == N_ACTION_CHUNKS_PER_STATE, (
#             f"Expected {N_ACTION_CHUNKS_PER_STATE} samples, "
#             f"but generated {write_idx}"
#         )

#         num_eps = N_ACTION_CHUNKS_PER_STATE

#         out_ep_len = np.full(
#             N_ACTION_CHUNKS_PER_STATE,
#             2,
#             dtype=np.int32,
#         )

#         out_ep_offset = np.arange(
#             0,
#             2 * N_ACTION_CHUNKS_PER_STATE,
#             2,
#             dtype=np.int64,
#         )

#         out_ep_idx = np.repeat(
#             np.arange(N_ACTION_CHUNKS_PER_STATE),
#             2,
#         ).astype(np.int32)

#         f_out.create_dataset("ep_len", data=out_ep_len)
#         f_out.create_dataset("ep_offset", data=out_ep_offset)
#         f_out.create_dataset("ep_idx", data=out_ep_idx)


# print(f"Saved dataset to {TARGET_FILE}")
# with h5py.File(TARGET_FILE, "r") as f_chk:

#     print("\nValidation:")
#     print("pixels:", f_chk["pixels"].shape)
#     print("action:", f_chk["action"].shape)
#     print("qpos:", f_chk["qpos"].shape)
#     print("qvel:", f_chk["qvel"].shape)

#     print("ep_len[:5] =", f_chk["ep_len"][:5])
#     print("ep_offset[:5] =", f_chk["ep_offset"][:5])