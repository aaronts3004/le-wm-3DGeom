import pathlib
from collections import defaultdict

import gymnasium
import numpy as np
from absl import app, flags
from stable_worldmodel.data import dataset
from stable_worldmodel.envs.two_room import env
from tqdm import trange
import os
from PIL import Image
import imageio
import h5py
import matplotlib.pyplot as plt

os.environ["MUJOCO_GL"] = "egl"

import ogbench.manipspace  # noqa
from ogbench.manipspace.oracles.markov.button_markov import ButtonMarkovOracle
from ogbench.manipspace.oracles.markov.cube_markov import CubeMarkovOracle
from ogbench.manipspace.oracles.markov.drawer_markov import DrawerMarkovOracle
from ogbench.manipspace.oracles.markov.window_markov import WindowMarkovOracle
from ogbench.manipspace.oracles.plan.button_plan import ButtonPlanOracle
from ogbench.manipspace.oracles.plan.cube_plan import CubePlanOracle
from ogbench.manipspace.oracles.plan.drawer_plan import DrawerPlanOracle
from ogbench.manipspace.oracles.plan.window_plan import WindowPlanOracle

FLAGS = flags.FLAGS
NUM_SOURCES = 1
NUM_GOALS = 1000         # number of trajectories per source
ACTION_BLOCKS=2

DEPTH=False
DEPTH_NORM = "inverse"

TOTAL_EPISODES = NUM_SOURCES * NUM_GOALS
OUTPUT_DIR = f"visualizations/generate_trajectories/{NUM_SOURCES}_sources_{NUM_GOALS}_goals_{ACTION_BLOCKS}_actionBlocks_{'DEPTH_' + DEPTH_NORM if DEPTH else 'RGB'}/"
FRAME_SKIP=5
os.makedirs(OUTPUT_DIR, exist_ok=True)
MAX_STEPS_PER_TRAJECTORY=ACTION_BLOCKS * FRAME_SKIP


DATASET_NAME = f"/home/student/data/ogbench/cube_overfit_{NUM_SOURCES}_sources_{NUM_GOALS}_traj_{ACTION_BLOCKS}_actionBlocks_{'DEPTH_' + DEPTH_NORM if DEPTH else 'RGB'}.h5"

flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_string('env_name', 'cube-single-v0', 'Environment name.')
flags.DEFINE_string('dataset_type', 'play', 'Dataset type.')
flags.DEFINE_string('save_path', None, 'Save path.')
flags.DEFINE_float('noise', 0.1, 'Action noise level.')
flags.DEFINE_float('noise_smoothing', 0.5, 'Action noise smoothing level for PlanOracle.')
flags.DEFINE_float('min_norm', 0.4, 'Minimum action norm for MarkovOracle.')
flags.DEFINE_float('p_random_action', 0, 'Probability of selecting a random action.')
flags.DEFINE_integer('num_episodes', 1000, 'Number of episodes.')
flags.DEFINE_integer('max_episode_steps', MAX_STEPS_PER_TRAJECTORY, 'Number of steps.')


def main(_):
    assert FLAGS.dataset_type in ['play', 'noisy']
    # 'play': Use a non-Markovian oracle (PlanOracle) that follows a pre-computed plan.
    # 'noisy': Use a Markovian, closed-loop oracle (MarkovOracle) with Gaussian action noise.

    # Initialize environment.
    env = gymnasium.make(
        FLAGS.env_name,
        terminate_at_goal=False,
        mode='data_collection',
        max_episode_steps=FLAGS.max_episode_steps,
        visualize_info=False, 
        width=224,
        height=224,
    )
    print(env.unwrapped._visualize_info)
    print(FLAGS)

    # Initialize oracles.
    oracle_type = 'plan' if FLAGS.dataset_type == 'play' else 'markov'
    has_button_states = hasattr(env.unwrapped, '_cur_button_states')
    if 'cube' in FLAGS.env_name:
        if oracle_type == 'markov':
            agents = {
                'cube': CubeMarkovOracle(env=env, min_norm=FLAGS.min_norm),
            }
        else:
            print("\n\n CubePlanOracle \n\n")
            agents = {
                'cube': CubePlanOracle(env=env, noise=FLAGS.noise, noise_smoothing=FLAGS.noise_smoothing),
            }
    elif 'scene' in FLAGS.env_name:
        if oracle_type == 'markov':
            agents = {
                'cube': CubeMarkovOracle(env=env, min_norm=FLAGS.min_norm, max_step=100),
                'button': ButtonMarkovOracle(env=env, min_norm=FLAGS.min_norm),
                'drawer': DrawerMarkovOracle(env=env, min_norm=FLAGS.min_norm),
                'window': WindowMarkovOracle(env=env, min_norm=FLAGS.min_norm),
            }
        else:
            agents = {
                'cube': CubePlanOracle(env=env, noise=FLAGS.noise, noise_smoothing=FLAGS.noise_smoothing),
                'button': ButtonPlanOracle(env=env, noise=FLAGS.noise, noise_smoothing=FLAGS.noise_smoothing),
                'drawer': DrawerPlanOracle(env=env, noise=FLAGS.noise, noise_smoothing=FLAGS.noise_smoothing),
                'window': WindowPlanOracle(env=env, noise=FLAGS.noise, noise_smoothing=FLAGS.noise_smoothing),
            }
    elif 'puzzle' in FLAGS.env_name:
        if oracle_type == 'markov':
            agents = {
                'button': ButtonMarkovOracle(env=env, min_norm=FLAGS.min_norm, gripper_always_closed=True),
            }
        else:
            agents = {
                'button': ButtonPlanOracle(
                    env=env,
                    noise=FLAGS.noise,
                    noise_smoothing=FLAGS.noise_smoothing,
                    gripper_always_closed=True,
                ),
            }

    # Collect data.
    dataset = defaultdict(list)
    total_steps = 0
    total_train_steps = 0
    num_train_episodes = FLAGS.num_episodes
    num_val_episodes = FLAGS.num_episodes // 10

    
    source_ids = []
    goal_ids = []
    episode_lengths = []
    episode_offsets = []
    pixels = []
    current_offset = 0
    dataset["qpos"] = []
    dataset["qvel"] = []


    src_idx = 0
    src_qpos = np.array([-1.8570e+00, -1.5349e+00,  2.1462e+00, -2.2110e+00, -1.5708e+00,
            2.7199e+00,  4.4934e-01,  5.8575e-04,  4.4669e-01, -4.4292e-01,
            4.4929e-01,  2.9956e-04,  4.4632e-01, -4.3992e-01,  4.7053e-01,
            4.2916e-03,  1.0841e-01, -7.5304e-01, -1.4355e-02,  1.7852e-02,
            6.5758e-01])

    src_qvel = np.array([ 7.3992e-02,  4.4032e-01,  4.0368e-01, -6.6603e-01,  1.0991e-03,
            1.5137e-01, -4.6606e-04,  1.4755e-04,  6.1730e-04, -1.4372e-03,
            -7.2293e-04, -7.0954e-04, -2.5147e-03,  6.0789e-03, -5.2453e-02,
            4.6213e-02, -2.9620e-01, -1.9350e-01,  7.8925e-02, -7.7470e-02])

    sanity_checks = 0
    for ep_idx in trange(TOTAL_EPISODES):
        # Have an additional while loop to handle rare cases with undesirable states (for the Scene environment).
        print("\n\nGenerating ep_idx=", ep_idx)
        ep_frames = []
        while True:
            ob, info = env.reset()
            env.unwrapped.set_state(src_qpos, src_qvel)
            img = env.unwrapped.render(camera="front_pixels", depth=DEPTH)

            if DEPTH: 
                if DEPTH_NORM == "linear":
                    img = np.clip(img, 0.5, 3.0)
                    img = (img-0.5)/2.5
                elif DEPTH_NORM == "inverse": 
                    img = np.clip(img,0.5,3.0)
                    img = 1/img
                    img = (img-img.min())/(img.max()-img.min())
                else: 
                    print("Unsopported DEPTH NORM, re-try")
                    exit(1)

                img = (img * 255).astype(np.uint8)
                img = np.repeat(img[..., None], 3, axis=-1)


            ep_frames.append(img)
            dataset["pixels"].append(img)
            dataset["qpos"].append(src_qpos.copy())
            dataset["qvel"].append(src_qvel.copy())

            if sanity_checks < 10: 
                img = env.unwrapped.render(camera="front_pixels", depth=DEPTH)

                if DEPTH: 
                    if DEPTH_NORM == "linear":
                        img = np.clip(img, 0.5, 3.0)
                        img = (img-0.5)/2.5
                    elif DEPTH_NORM == "inverse": 
                        img = np.clip(img,0.5,3.0)
                        img = 1/img
                        img = (img-img.min())/(img.max()-img.min())
                    else: 
                        print("Unsopported DEPTH NORM, re-try")
                        exit(1)

                    print("img.shape=", img.shape)
                    plt.imshow(img, cmap="viridis")
                    plt.colorbar(label="Depth (m)")
                    plt.savefig(os.path.join(OUTPUT_DIR, f"ep_idx_{ep_idx}_source_frame_DEPTH.png"), dpi=200, bbox_inches="tight")
                    plt.close()

                    img = (img * 255).astype(np.uint8)
                    img = np.repeat(img[..., None], 3, axis=-1)

                    plt.imshow(img)
                    plt.savefig(os.path.join(OUTPUT_DIR, f"ep_idx_{ep_idx}_source_frame_DEPTH_WM_INPUT.png"), dpi=200, bbox_inches="tight")
                    plt.close()
                else: 
                    print("DEPTH =", DEPTH)
                    print("img.dtype =", img.dtype)
                    print("img.shape =", img.shape)
                    print("img.min(), img.max() =", img.min(), img.max())
                    Image.fromarray(img).save(os.path.join(OUTPUT_DIR, f"ep_idx_{ep_idx}_source_frame.png"))

            # Set the cube stacking probability for this episode.
            if 'single' in FLAGS.env_name:                      
                p_stack = 0.0
            elif 'double' in FLAGS.env_name:
                p_stack = np.random.uniform(0.0, 0.25)
            elif 'triple' in FLAGS.env_name:
                p_stack = np.random.uniform(0.05, 0.35)
            elif 'quadruple' in FLAGS.env_name:
                p_stack = np.random.uniform(0.1, 0.5)
            elif 'octuple' in FLAGS.env_name:
                p_stack = np.random.uniform(0.0, 0.35)
            else:
                p_stack = 0.5

            if oracle_type == 'markov':
                # Set the action noise level for this episode.
                xi = np.random.uniform(0, FLAGS.noise)

            agent = agents[info['privileged/target_task']]
            agent.reset(ob, info)

            done = False
            step = 0

            action_block = []
            while not done:

                if np.random.rand() < FLAGS.p_random_action:
                    # Sample a random action.
                    action = env.action_space.sample()
                else:
                    # Get an action from the oracle.
                    action = agent.select_action(ob, info)
                    action = np.array(action)
                    if oracle_type == 'markov':
                        # Add Gaussian noise to the action.
                        action = action + np.random.normal(0, [xi, xi, xi, xi * 3, xi * 10], action.shape)
                action = np.clip(action, -1, 1)
                next_ob, reward, terminated, truncated, info = env.step(action)
                action_block.append(action)

                if (step + 1) % FRAME_SKIP == 0: 

                    img = env.unwrapped.render(camera="front_pixels", depth=DEPTH)

                    if DEPTH: 
                        if DEPTH_NORM == "linear":
                            img = np.clip(img, 0.5, 3.0)
                            img = (img-0.5)/2.5
                        elif DEPTH_NORM == "inverse": 
                            img = np.clip(img,0.5,3.0)
                            img = 1/img
                            img = (img-img.min())/(img.max()-img.min())
                        else: 
                            print("Unsopported DEPTH NORM, re-try")
                            exit(1)

                        img = (img * 255).astype(np.uint8)
                        img = np.repeat(img[..., None], 3, axis=-1)
                    
                    ep_frames.append(img)

                    dataset['pixels'].append(img)
                    dataset["action"].append(
                        np.concatenate(action_block, axis=0)
                    )
                    dataset["qpos"].append(env.unwrapped.data.qpos.copy())          
                    dataset["qvel"].append(env.unwrapped.data.qvel.copy())
                    action_block.clear()


                if ep_idx == 0 and step % 1 == 0: 
                    img = env.unwrapped.render(camera="front_pixels", depth=DEPTH)

                    if DEPTH: 
                        if DEPTH_NORM == "linear":
                            img = np.clip(img, 0.5, 3.0)
                            img = (img-0.5)/2.5
                        elif DEPTH_NORM == "inverse": 
                            img = np.clip(img,0.5,3.0)
                            img = 1/img
                            img = (img-img.min())/(img.max()-img.min())
                        else: 
                            print("Unsopported DEPTH NORM, re-try")
                            exit(1)

                        plt.imshow(img, cmap="viridis")
                        plt.colorbar(label="Depth (m)")
                        plt.savefig(os.path.join(OUTPUT_DIR, f"ep_idx_{ep_idx}_frame_{step}_DEPTH.png"), dpi=200, bbox_inches="tight")
                        plt.close()

                        img = (img * 255).astype(np.uint8)
                        img = np.repeat(img[..., None], 3, axis=-1)

                        plt.imshow(img)
                        plt.savefig(os.path.join(OUTPUT_DIR, f"ep_idx_{ep_idx}_frame_{step}_DEPTH_WM_INPUT.png"), dpi=200, bbox_inches="tight")
                        plt.close()



                    else: 
                        Image.fromarray(img).save(os.path.join(OUTPUT_DIR, f"ep_idx_{ep_idx}_frame_{step}.png"))

                done = terminated or truncated

                if agent.done:
                    # Set a new task when the current task is done.
                    print(f"agent done={agent.done}, step={step}, terminated={terminated}, truncated={truncated}")
                    agent_ob, agent_info = env.unwrapped.set_new_target(p_stack=p_stack)
                    agent = agents[agent_info['privileged/target_task']]
                    agent.reset(agent_ob, agent_info)

                # dataset['observations'].append(ob)
                # dataset['qpos'].append(info['prev_qpos'])
                # dataset['qvel'].append(info['prev_qvel'])

                if has_button_states:
                    dataset['button_states'].append(info['prev_button_states'])

                ob = next_ob
                step += 1

            if 'scene' in FLAGS.env_name:
                print("Environment is a SCENE")
                # Perform health check. We want to ensure that the cube is always visible unless it's in the drawer.
                # Otherwise, the test-time goal images may become ambiguous.
                is_healthy = True
                ep_qpos = np.array(ep_qpos)
                block_xyzs = ep_qpos[:, 14:17]
                if (block_xyzs[:, 1] >= 0.29).any():
                    is_healthy = False  # Block goes too far right.
                if ((block_xyzs[:, 1] <= -0.3) & ((block_xyzs[:, 2] < 0.06) | (block_xyzs[:, 2] > 0.08))).any():
                    is_healthy = False  # Block goes too far left, without being in the drawer.

                if is_healthy:
                    break
                else:
                    # Remove the last episode and retry.
                    print('Unhealthy episode, retrying...', flush=True)
                    for k in dataset.keys():
                        dataset[k] = dataset[k][:-step]
            else:
                print("done with generating ep_idx=", ep_idx)

                dataset["action"].append(
                    np.zeros_like(dataset["action"][-1])
                )


                episode_lengths.append(len(ep_frames))
                episode_offsets.append(current_offset)
                source_ids.append(src_idx)
                goal_ids.append(ep_idx)
                current_offset += len(ep_frames)

                sanity_checks += 1
                # if sanity_checks < 10:
                #     output_path = os.path.join(OUTPUT_DIR, f"full_{ep_idx}.gif")
                #     imageio.mimsave(output_path, ep_frames, fps=5)
                #     sanity_checks += 1

                #     print(f"Saved gif with {len(ep_frames)} frames in total !")
                
                break


        total_steps += step
        print(f"total_Steps= {total_steps} + ***********")
        if ep_idx < num_train_episodes:
            total_train_steps += step

    print('Total steps:', total_steps)
    print("\n Saved visualizations to: ", OUTPUT_DIR)


    print("\n Creating target dataset")
    with h5py.File(DATASET_NAME, "w") as f:

        f.create_dataset(
            "source_id",
            data=np.asarray(source_ids, dtype=np.int32),
        )

        f.create_dataset(
            "goal_id",
            data=np.asarray(goal_ids, dtype=np.int32),
        )

        f.create_dataset(
            "observations",
            data=np.asarray(dataset["observations"]),
            compression="gzip",
        )

        f.create_dataset(
            "pixels",
            data=np.asarray(dataset["pixels"], dtype=np.uint8),
            compression="gzip",
        )

        f.create_dataset(
            "action",
            data=np.asarray(dataset["action"], dtype=np.float32),
            compression="gzip",
        )

        f.create_dataset(
            "qpos",
            data=np.asarray(dataset["qpos"], dtype=np.float32),
            compression="gzip",
        )

        f.create_dataset(
            "qvel",
            data=np.asarray(dataset["qvel"], dtype=np.float32),
            compression="gzip",
        )

        f.create_dataset(
            "ep_len",
            data=np.asarray(episode_lengths, dtype=np.int32),
        )

        f.create_dataset(
            "ep_offset",
            data=np.asarray(episode_offsets, dtype=np.int64),
        )

        print("Created target dataset!\n Wrote to: ", DATASET_NAME)


    print("Sanity check on the HDF5 structure")
    f = h5py.File(DATASET_NAME, "r")

    print("pixels:", f["pixels"].shape)
    print("action:", f["action"].shape)
    print("ep_len[:10]:", f["ep_len"][:10])
    print("ep_offset[:10]:", f["ep_offset"][:10])

    img = f["pixels"][0]
    Image.fromarray(img).save(f"{OUTPUT_DIR}/sanity_check_first_frame_from_dataset.png")


if __name__ == '__main__':
    app.run(main)

    
