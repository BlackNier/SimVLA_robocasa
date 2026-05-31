"""
RoboCasa LeRobot data handler for SimVLA.

This adapter reads RoboCasa365 v1.0 datasets in LeRobot format and emits the
sample structure expected by datasets.dataset_smolvlm.SmolVLMDataReader:

    language_instruction: str
    image_input: FloatTensor[V, C, H, W]
    image_mask: BoolTensor[V]
    proprio: FloatTensor[16]
    abs_trajectory: FloatTensor[num_actions + 1, 12]

State order is kept identical between training and evaluation:
    eef_pos_rel(3), eef_rot_rel_quat(4), base_pos(3), base_rot_quat(4), gripper_qpos(2)

Action order matches robocasa.utils.env_utils.convert_action:
    eef_pos(3), eef_rot_axis_angle(3), gripper_close(1), base_motion(4), control_mode(1)
"""

from __future__ import annotations

import copy
import random
from functools import lru_cache
from typing import Iterable

import numpy as np
import torch
from PIL import Image

from .base import DomainHandler


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

ROBOCASA_ACTION_KEYS = [
    "action.end_effector_position",
    "action.end_effector_rotation",
    "action.gripper_close",
    "action.base_motion",
    "action.control_mode",
]

ROBOCASA_LANGUAGE_KEY = "annotation.human.task_description"


def make_robocasa_modality_configs(num_actions: int):
    from robocasa.utils.groot_utils.groot_dataset import ModalityConfig

    return {
        "video": ModalityConfig(
            delta_indices=[0],
            modality_keys=ROBOCASA_VIDEO_KEYS,
        ),
        "state": ModalityConfig(
            delta_indices=[0],
            modality_keys=ROBOCASA_STATE_KEYS,
        ),
        "action": ModalityConfig(
            delta_indices=list(range(num_actions + 1)),
            modality_keys=ROBOCASA_ACTION_KEYS,
        ),
        "language": ModalityConfig(
            delta_indices=[0],
            modality_keys=[ROBOCASA_LANGUAGE_KEY],
        ),
    }


def _as_float32_2d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        x = x[None]
    return x


def concat_robocasa_state(step: dict) -> np.ndarray:
    parts = [_as_float32_2d(step[k]) for k in ROBOCASA_STATE_KEYS]
    state = np.concatenate(parts, axis=-1)
    return state[0].astype(np.float32)


def concat_robocasa_action(step: dict) -> np.ndarray:
    parts = [_as_float32_2d(step[k]) for k in ROBOCASA_ACTION_KEYS]
    return np.concatenate(parts, axis=-1).astype(np.float32)


def extract_robocasa_language(step: dict, fallback: str = "") -> str:
    value = step.get(ROBOCASA_LANGUAGE_KEY, fallback)
    if isinstance(value, (list, tuple)):
        value = value[0] if value else fallback
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    return str(value)


def image_from_robocasa_frame(frame: np.ndarray) -> Image.Image:
    frame = np.asarray(frame)
    if frame.ndim == 4:
        frame = frame[0]
    if frame.dtype != np.uint8:
        if np.issubdtype(frame.dtype, np.floating):
            frame = np.clip(frame * 255.0, 0, 255).astype(np.uint8)
        else:
            frame = frame.astype(np.uint8)
    return Image.fromarray(frame)


@lru_cache(maxsize=128)
def _make_single_dataset(dataset_path: str, filter_key: str | None, num_actions: int):
    from robocasa.utils.groot_utils.groot_dataset import LeRobotSingleDataset
    from robocasa.utils.groot_utils.schema import EmbodimentTag

    return LeRobotSingleDataset(
        dataset_path=dataset_path,
        modality_configs=make_robocasa_modality_configs(num_actions),
        embodiment_tag=EmbodimentTag("new_embodiment"),
        filter_key=filter_key,
    )


class RobocasaLeRobotHandler(DomainHandler):
    dataset_name = "robocasa_lerobot"

    def __init__(self, meta: dict, num_views: int = 3) -> None:
        super().__init__(meta, num_views)
        self.datalist = meta["datalist"]

    def _dataset_for(self, traj_idx: int, num_actions: int):
        ds_meta = self.datalist[traj_idx]
        return _make_single_dataset(
            str(ds_meta["path"]),
            ds_meta.get("filter_key"),
            int(num_actions),
        )

    def iter_episode(
        self,
        traj_idx: int,
        *,
        num_actions: int = 16,
        training: bool = True,
        image_aug=None,
        action_mode: str = "robocasa_12dof",
        lang_aug_map: dict | None = None,
        **kwargs,
    ) -> Iterable[dict]:
        dataset = self._dataset_for(traj_idx, num_actions)
        indices = list(range(len(dataset)))
        if training:
            random.shuffle(indices)

        image_mask = torch.zeros(self.num_views, dtype=torch.bool)
        image_mask[: min(self.num_views, len(ROBOCASA_VIDEO_KEYS))] = True

        for index in indices:
            try:
                step = dataset[index]

                instruction = extract_robocasa_language(
                    step,
                    fallback=self.datalist[traj_idx].get("task", ""),
                )
                if training and lang_aug_map and instruction in lang_aug_map:
                    instruction = random.choice(lang_aug_map[instruction])

                imgs = []
                for key in ROBOCASA_VIDEO_KEYS[: self.num_views]:
                    img = image_from_robocasa_frame(step[key])
                    imgs.append(image_aug(img) if image_aug else img)
                while len(imgs) < self.num_views:
                    imgs.append(torch.zeros_like(imgs[0]))

                yield {
                    "language_instruction": instruction,
                    "image_input": torch.stack(imgs, dim=0),
                    "image_mask": image_mask,
                    "proprio": torch.tensor(concat_robocasa_state(step), dtype=torch.float32),
                    "abs_trajectory": torch.tensor(concat_robocasa_action(step), dtype=torch.float32),
                }
            except Exception as exc:
                if not training:
                    raise
                print(f"[RobocasaLeRobotHandler] skipped sample {traj_idx}/{index}: {exc}")
                continue


def create_robocasa_meta(
    dataset_soup: str,
    output_path: str,
    dataset_base_path: str | None = None,
    split: str = "target",
    source: str = "human",
    demo_fraction: float = 1.0,
) -> dict:
    if dataset_base_path:
        import robocasa.macros as macros

        macros.DATASET_BASE_PATH = dataset_base_path

    from robocasa.utils.dataset_registry import DATASET_SOUP_REGISTRY
    from robocasa.utils.dataset_registry_utils import get_ds_soup

    if dataset_soup in DATASET_SOUP_REGISTRY:
        datalist = copy.deepcopy(DATASET_SOUP_REGISTRY[dataset_soup])
    else:
        task_set = dataset_soup
        if dataset_soup.startswith("target_"):
            task_set = dataset_soup.removeprefix("target_")
        datalist = get_ds_soup(
            split=split,
            task_set=task_set,
            source=source,
            demo_fraction=demo_fraction,
        )

    meta = {
        "dataset_name": "robocasa_lerobot",
        "dataset_soup": dataset_soup,
        "datalist": datalist,
        "num_datasets": len(datalist),
        "state_dim": 16,
        "action_dim": 12,
        "num_views": 3,
        "fps": 20,
        "state_keys": ROBOCASA_STATE_KEYS,
        "action_keys": ROBOCASA_ACTION_KEYS,
        "video_keys": ROBOCASA_VIDEO_KEYS,
        "language_key": ROBOCASA_LANGUAGE_KEY,
    }

    import json

    with open(output_path, "w") as f:
        json.dump(meta, f, indent=2)
    return meta
