import os

os.environ["MUJOCO_GL"] = "egl"

import time
from pathlib import Path

import hydra
import numpy as np
import stable_pretraining as spt
import torch
from omegaconf import DictConfig, OmegaConf
from sklearn import preprocessing
from torchvision.transforms import v2 as transforms
import stable_worldmodel as swm
from preprocessing_eval import normal_transform

def img_transform(cfg):
    transform = transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(**spt.data.dataset_stats.ImageNet),
            transforms.Resize(size=cfg.eval.img_size),
        ]
    )
    return transform


def get_episodes_length(dataset, episodes):
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"

    episode_idx = dataset.get_col_data(col_name)
    step_idx = dataset.get_col_data("step_idx")
    lengths = []
    for ep_id in episodes:
        lengths.append(np.max(step_idx[episode_idx == ep_id]) + 1)
    return np.array(lengths)


def get_dataset(cfg, dataset_name):
    dataset_path = Path(cfg.cache_dir or swm.data.utils.get_cache_dir())
    dataset = swm.data.HDF5Dataset(
        dataset_name,
        keys_to_cache=cfg.dataset.keys_to_cache,
        cache_dir=dataset_path,
    )
    return dataset



'''

TODO:
- Run with: python eval.py --config-name=cube_RGB.yaml (or other yaml launch files)

To edit a yaml launch file:
- Set the dataset path in eval.dataset_name
- Set the ckpt path in policy (without the "_object.ckpt" extension !)

Other metrics (+ default value): 
- eval.goal_offset_steps: (25) at which world-model-timestep does the goal happen
- plan_config.horizon (5): CEM solver block size (executing inside imagination / CEM solver)
- plan_config.receding_horizon (5): number of actions actually executed before re-planning; is equal to frameskip
- eval.eval_budget (50): execute maximum 50 real actions, if goal not reached until then, then is failure; 
                         even if the goal was not reached at (goal_offset_steps), the WM can still recover and reach it later


- eval.num_eval (50): number of episodes to rollout


'''




@hydra.main(version_base=None, config_path="./config/eval", config_name="pusht")
def run(cfg: DictConfig):
    """Run evaluation of dinowm vs random policy."""

    # by default, horizon=5,  action_block=5 = frameskip, eval_budget=50
    # horizon     ->  size of one planning block, for one new planning step how long to "imagine"
    # eval_budget ->  for how many real environment steps is the model allowed to execute
    assert (
        cfg.plan_config.horizon * cfg.plan_config.action_block <= cfg.eval.eval_budget
    ), "Planning horizon must be smaller than or equal to eval_budget"

    # create world environment
    cfg.world.max_episode_steps = 2 * cfg.eval.eval_budget
    world = swm.World(**cfg.world, image_shape=(224, 224))

    print("Running eval with the following config:")
    print("horizon=", cfg.plan_config.horizon)
    print("action block=", cfg.plan_config.action_block)
    print("eval budget=", cfg.eval.eval_budget)

    # create the transform
    if cfg.dataset.modality == "RGB": 
        transform = {"pixels": img_transform(cfg),"goal": img_transform(cfg),}
    elif cfg.dataset.modality == "NRM": 
        transform = {"pixels": normal_transform(cfg),"goal": normal_transform(cfg),}
    else: 
        print("Unsupported modality for transform")
        exit(1)

    dataset = get_dataset(cfg, cfg.eval.dataset_name)

    stats_dataset = dataset  # get_dataset(cfg, cfg.dataset.stats)
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    ep_indices, _ = np.unique(stats_dataset.get_col_data(col_name), return_index=True)

    for idx in ep_indices[:5]:
        print("idx =", idx)
        row = dataset.get_row_data([idx])
        print(type(row))

    print("number of unique episodes in dataset: ", ep_indices.shape)



    process = {}
    for col in cfg.dataset.keys_to_cache:
        if col in ["pixels"]:
            continue
        processor = preprocessing.StandardScaler()
        col_data = stats_dataset.get_col_data(col)
        col_data = col_data[~np.isnan(col_data).any(axis=1)]
        processor.fit(col_data)
        process[col] = processor

        if col != "action":
            process[f"goal_{col}"] = process[col]

    # -- run evaluation
    policy = cfg.get("policy", "random")

    if policy != "random":
        model = swm.policy.AutoCostModel(cfg.policy)
        model = model.to("cuda")
        model = model.eval()
        model.requires_grad_(False)
        model.interpolate_pos_encoding = True
        config = swm.PlanConfig(**cfg.plan_config)
        solver = hydra.utils.instantiate(cfg.solver, model=model)
        policy = swm.policy.WorldModelPolicy(
            solver=solver, config=config, process=process, transform=transform
        )

    else:

        print("Running with random policy ! Choose a different cfg.policy")
        exit(1)

        policy = swm.policy.RandomPolicy()

    results_path = (
        Path(swm.data.utils.get_cache_dir(), cfg.policy).parent
        if cfg.policy != "random"
        else Path(__file__).parent
    )

    # sample the episodes and the starting indices
    episode_len = get_episodes_length(dataset, ep_indices)
    max_start_idx = episode_len - cfg.eval.goal_offset_steps - 1
    max_start_idx_dict = {ep_id: max_start_idx[i] for i, ep_id in enumerate(ep_indices)}
    # Map each dataset row’s episode_idx to its max_start_idx
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    max_start_per_row = np.array(
        [max_start_idx_dict[ep_id] for ep_id in dataset.get_col_data(col_name)]
    )

    # remove all the lines of dataset for which dataset['step_idx'] > max_start_per_row
    valid_mask = dataset.get_col_data("step_idx") <= max_start_per_row
    ep_idx = dataset.get_col_data("ep_idx")

    if getattr(dataset, "orig_to_internal", None) is not None:
        mapper = np.vectorize(dataset.orig_to_internal.__getitem__)
        ep_idx = mapper(ep_idx)

    valid_mask &= (
        ep_idx >= cfg.eval.first_episode
    ) & (
        ep_idx <= cfg.eval.last_episode
    )

    valid_indices = np.nonzero(valid_mask)[0]
    print(valid_mask.sum(), "valid starting points found for evaluation.")

    print("Unique internal episodes:",np.unique(ep_idx[valid_mask]))

    # valid_indices = np.nonzero(valid_mask)[0]
    # print(valid_mask.sum(), "valid starting points found for evaluation.")

    g = np.random.default_rng(cfg.seed)
    random_episode_indices = g.choice(
        len(valid_indices) - 1, size=cfg.eval.num_eval, replace=False
    )

    # sort increasingly to avoid issues with HDF5Dataset indexing
    random_episode_indices = np.sort(valid_indices[random_episode_indices])

    print(random_episode_indices)

    eval_episodes = dataset.get_row_data(random_episode_indices)[col_name]
    eval_start_idx = dataset.get_row_data(random_episode_indices)["step_idx"]

    print("eval_episodes min:", eval_episodes.min())
    print("eval_episodes max:", eval_episodes.max())
    print("num episodes:", len(dataset.offsets))
    print(dataset.get_col_data("ep_idx")[:20])
    print(dataset.get_col_data("ep_idx").max())

    print("\n\nALL EVAL EPISODE INDICES: ", eval_episodes)

    print("\n\nNumber of episodes to rollout = ", len(eval_episodes))

    if len(eval_episodes) < cfg.eval.num_eval:
        raise ValueError("Not enough episodes with sufficient length for evaluation.")

    world.set_policy(policy)

    start_time = time.time()
    metrics = world.evaluate(
        dataset=dataset,
        episodes_idx=eval_episodes.tolist(),
        start_steps=eval_start_idx.tolist(),
        goal_offset=cfg.eval.goal_offset_steps,
        eval_budget=cfg.eval.eval_budget,
        callables=OmegaConf.to_container(cfg.eval.get("callables"), resolve=True),
        video=results_path,
    )
    end_time = time.time()
    
    print(metrics)

    results_path = results_path / cfg.output.filename
    results_path.parent.mkdir(parents=True, exist_ok=True)

    print("Writing results to: ", results_path)

    with results_path.open("a") as f:
        f.write("\n")  # separate from previous runs

        f.write("==== CONFIG ====\n")
        f.write(OmegaConf.to_yaml(cfg))
        f.write("\n")

        f.write("==== RESULTS ====\n")
        f.write(f"metrics: {metrics}\n")
        f.write(f"evaluation_time: {end_time - start_time} seconds\n")


if __name__ == "__main__":
    run()
