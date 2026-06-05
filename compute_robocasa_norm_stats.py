#!/usr/bin/env python3
"""Compute SimVLA normalization statistics for RoboCasa365 LeRobot datasets."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import random
import sys

import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datasets.domain_handler.robocasa_lerobot import (
    concat_robocasa_action,
    concat_robocasa_state,
    ROBOCASA_ACTION_KEYS,
    ROBOCASA_STATE_KEYS,
)


def make_robocasa_norm_modality_configs(num_actions: int):
    from robocasa.utils.groot_utils.groot_dataset import ModalityConfig

    return {
        "state": ModalityConfig(
            delta_indices=[0],
            modality_keys=ROBOCASA_STATE_KEYS,
        ),
        "action": ModalityConfig(
            delta_indices=list(range(num_actions)),
            modality_keys=ROBOCASA_ACTION_KEYS,
        ),
    }


def _load_single_dataset(dataset_path: str, filter_key: str | None, num_actions: int):
    from robocasa.utils.groot_utils.groot_dataset import LeRobotSingleDataset
    from robocasa.utils.groot_utils.schema import EmbodimentTag

    return LeRobotSingleDataset(
        dataset_path=dataset_path,
        modality_configs=make_robocasa_norm_modality_configs(num_actions),
        embodiment_tag=EmbodimentTag("new_embodiment"),
        video_backend=os.environ.get("ROBOCASA_VIDEO_BACKEND", "opencv"),
        filter_key=filter_key,
    )


def _stats(x: np.ndarray) -> dict:
    return {
        "mean": np.mean(x, axis=0).tolist(),
        "std": (np.std(x, axis=0) + 1e-6).tolist(),
        "min": np.min(x, axis=0).tolist(),
        "max": np.max(x, axis=0).tolist(),
        "q01": np.quantile(x, 0.01, axis=0).tolist(),
        "q99": np.quantile(x, 0.99, axis=0).tolist(),
    }


def _collect_dataset_stats(job: tuple[int, dict, int, int, int]) -> tuple[np.ndarray, np.ndarray]:
    ds_index, ds_meta, num_actions, max_samples_per_dataset, seed = job
    dataset_path = ds_meta["path"]
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(dataset_path)

    dataset = _load_single_dataset(
        dataset_path=dataset_path,
        filter_key=ds_meta.get("filter_key"),
        num_actions=num_actions,
    )

    indices = list(range(len(dataset)))
    if max_samples_per_dataset and len(indices) > max_samples_per_dataset:
        rng = random.Random(seed + ds_index)
        indices = rng.sample(indices, max_samples_per_dataset)

    states = []
    actions = []
    for idx in indices:
        step = dataset[idx]
        states.append(concat_robocasa_state(step))
        actions.append(concat_robocasa_action(step).reshape(-1, 12))

    return (
        np.asarray(states, dtype=np.float32),
        np.concatenate(actions, axis=0).astype(np.float32),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metas_path", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--num_actions", type=int, default=16)
    parser.add_argument("--max_samples_per_dataset", type=int, default=20000)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    with open(args.metas_path, "r") as f:
        meta = json.load(f)

    random.seed(args.seed)
    np.random.seed(args.seed)

    jobs = [
        (i, ds_meta, args.num_actions, args.max_samples_per_dataset, args.seed)
        for i, ds_meta in enumerate(meta["datalist"])
    ]

    states = []
    actions = []
    if args.num_workers <= 1:
        iterator = map(_collect_dataset_stats, jobs)
        for state_chunk, action_chunk in tqdm(iterator, total=len(jobs), desc="datasets"):
            states.append(state_chunk)
            actions.append(action_chunk)
    else:
        with mp.get_context("spawn").Pool(processes=args.num_workers) as pool:
            iterator = pool.imap_unordered(_collect_dataset_stats, jobs)
            for state_chunk, action_chunk in tqdm(iterator, total=len(jobs), desc="datasets"):
                states.append(state_chunk)
                actions.append(action_chunk)

    state_arr = np.concatenate(states, axis=0).astype(np.float32)
    action_arr = np.concatenate(actions, axis=0).astype(np.float32)

    result = {
        "norm_stats": {
            "state": _stats(state_arr),
            "actions": _stats(action_arr),
            "metadata": {
                "dataset": "robocasa_lerobot",
                "state_dim": int(state_arr.shape[-1]),
                "action_dim": int(action_arr.shape[-1]),
                "num_state_samples": int(state_arr.shape[0]),
                "num_action_samples": int(action_arr.shape[0]),
            },
        }
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved norm stats to {args.output}")


if __name__ == "__main__":
    main()
