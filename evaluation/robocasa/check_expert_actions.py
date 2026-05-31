#!/usr/bin/env python3
"""Sanity-check RoboCasa expert actions before evaluating a learned policy."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

ROBOCASA_ACTION_KEYS = [
    "action.end_effector_position",
    "action.end_effector_rotation",
    "action.gripper_close",
    "action.base_motion",
    "action.control_mode",
]

ROBOCASA_VIDEO_KEYS = [
    "video.robot0_agentview_left",
    "video.robot0_eye_in_hand",
    "video.robot0_agentview_right",
]

ROBOCASA_STATE_KEYS = [
    "state.end_effector_position_relative",
    "state.end_effector_rotation_relative",
    "state.base_position",
    "state.base_rotation",
    "state.gripper_qpos",
]

ROBOCASA_LANGUAGE_KEY = "annotation.human.task_description"


def make_robocasa_modality_configs(num_actions: int):
    from robocasa.utils.groot_utils.groot_dataset import ModalityConfig

    return {
        "video": ModalityConfig(delta_indices=[0], modality_keys=ROBOCASA_VIDEO_KEYS),
        "state": ModalityConfig(delta_indices=[0], modality_keys=ROBOCASA_STATE_KEYS),
        "action": ModalityConfig(
            delta_indices=list(range(num_actions + 1)),
            modality_keys=ROBOCASA_ACTION_KEYS,
        ),
        "language": ModalityConfig(
            delta_indices=[0],
            modality_keys=[ROBOCASA_LANGUAGE_KEY],
        ),
    }


def concat_robocasa_action(step: dict) -> np.ndarray:
    parts = []
    for key in ROBOCASA_ACTION_KEYS:
        x = np.asarray(step[key], dtype=np.float32)
        if x.ndim == 1:
            x = x[None]
        parts.append(x)
    return np.concatenate(parts, axis=-1).astype(np.float32)

def _resolve_dataset_path(dataset_base_path: str, task: str, split: str, source: str) -> str:
    import robocasa.macros as macros
    from robocasa.utils.dataset_registry_utils import get_ds_meta

    macros.DATASET_BASE_PATH = dataset_base_path
    meta = get_ds_meta(task=task, split=split, source=source)
    if meta is None:
        raise ValueError(f"No dataset registered for task={task}, split={split}, source={source}")
    return meta["path"]


def compare_adapter_actions(dataset_path: str, episode: int, num_actions: int):
    from robocasa.utils import lerobot_utils as LU
    from robocasa.utils.groot_utils.groot_dataset import LeRobotSingleDataset
    from robocasa.utils.groot_utils.schema import EmbodimentTag

    dataset_path_obj = Path(dataset_path)
    official = LU.get_episode_actions(dataset_path_obj, episode)

    dataset = LeRobotSingleDataset(
        dataset_path=dataset_path,
        modality_configs=make_robocasa_modality_configs(num_actions),
        embodiment_tag=EmbodimentTag("new_embodiment"),
        video_backend=os.environ.get("ROBOCASA_VIDEO_BACKEND", "opencv"),
        filter_key=None,
    )
    step = dataset.get_step_data(episode, 0)
    adapter = concat_robocasa_action(step)

    n = min(len(official), len(adapter))
    max_abs_err = float(np.max(np.abs(official[:n] - adapter[:n])))
    print(f"dataset_path: {dataset_path}")
    print(f"episode: {episode}")
    print(f"official action shape: {official.shape}")
    print(f"adapter action chunk shape: {adapter.shape}")
    print(f"max abs error on first {n} actions: {max_abs_err:.8f}")
    print(f"official first action: {official[0].tolist()}")
    print(f"adapter  first action: {adapter[0].tolist()}")
    if max_abs_err > 1e-5:
        raise RuntimeError("Adapter action order/value does not match RoboCasa official action.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default=None)
    parser.add_argument("--dataset_base_path", type=str, default=None)
    parser.add_argument("--task", type=str, default="CloseFridge")
    parser.add_argument("--split", type=str, default="target", choices=["pretrain", "target", "real"])
    parser.add_argument("--source", type=str, default="human")
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--num_actions", type=int, default=16)
    args = parser.parse_args()

    dataset_path = args.dataset_path
    if dataset_path is None:
        if args.dataset_base_path is None:
            raise ValueError("Pass either --dataset_path or --dataset_base_path.")
        dataset_path = _resolve_dataset_path(
            dataset_base_path=args.dataset_base_path,
            task=args.task,
            split=args.split,
            source=args.source,
        )

    compare_adapter_actions(dataset_path, args.episode, args.num_actions)


if __name__ == "__main__":
    main()
