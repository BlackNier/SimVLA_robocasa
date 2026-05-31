#!/usr/bin/env python3
"""Compare SimVLA predicted actions against RoboCasa expert actions offline."""

from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file as load_safetensors


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
        "video": ModalityConfig(delta_indices=[0], modality_keys=ROBOCASA_VIDEO_KEYS),
        "state": ModalityConfig(delta_indices=[0], modality_keys=ROBOCASA_STATE_KEYS),
        "action": ModalityConfig(
            delta_indices=list(range(num_actions)),
            modality_keys=ROBOCASA_ACTION_KEYS,
        ),
        "language": ModalityConfig(
            delta_indices=[0],
            modality_keys=[ROBOCASA_LANGUAGE_KEY],
        ),
    }


def _as_2d(x) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        x = x[None]
    return x


def concat_state(step: dict) -> np.ndarray:
    return np.concatenate([_as_2d(step[k]) for k in ROBOCASA_STATE_KEYS], axis=-1)[0]


def concat_action(step: dict) -> np.ndarray:
    return np.concatenate([_as_2d(step[k]) for k in ROBOCASA_ACTION_KEYS], axis=-1)


def get_language(step: dict) -> str:
    value = step[ROBOCASA_LANGUAGE_KEY]
    if isinstance(value, (list, tuple)):
        value = value[0]
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    return str(value)


def image_frame(step: dict, key: str) -> np.ndarray:
    frame = np.asarray(step[key])
    if frame.ndim == 4:
        frame = frame[0]
    if frame.dtype != np.uint8:
        if np.issubdtype(frame.dtype, np.floating):
            frame = np.clip(frame * 255.0, 0, 255).astype(np.uint8)
        else:
            frame = frame.astype(np.uint8)
    return np.ascontiguousarray(frame)


def resolve_dataset_path(dataset_base_path: str, task: str, split: str, source: str) -> str:
    import robocasa.macros as macros
    from robocasa.utils.dataset_registry_utils import get_ds_meta

    macros.DATASET_BASE_PATH = dataset_base_path
    meta = get_ds_meta(task=task, split=split, source=source)
    if meta is None:
        raise ValueError(f"No dataset registered for task={task}, split={split}, source={source}")
    return meta["path"]


def load_model(checkpoint: str, norm_stats: str | None, smolvlm_model: str, device: str):
    # Import SimVLA after robocasa/lerobot have had a chance to import the
    # HuggingFace `datasets` package. This avoids a name clash with SimVLA's
    # local `datasets/` directory.
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root))

    from models.configuration_smolvlm_vla import SmolVLMVLAConfig
    from models.modeling_smolvlm_vla import SmolVLMVLA
    from models.processing_smolvlm_vla import SmolVLMVLAProcessor

    config = SmolVLMVLAConfig.from_pretrained(checkpoint)
    if smolvlm_model:
        config.smolvlm_model_path = smolvlm_model

    model = SmolVLMVLA(config)
    safetensors_path = os.path.join(checkpoint, "model.safetensors")
    torch_path = os.path.join(checkpoint, "pytorch_model.bin")
    if os.path.exists(safetensors_path):
        state_dict = load_safetensors(safetensors_path, device="cpu")
    elif os.path.exists(torch_path):
        state_dict = torch.load(torch_path, map_location="cpu")
    else:
        raise FileNotFoundError(f"No model weights found in {checkpoint}")
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"loaded checkpoint; missing={len(missing)}, unexpected={len(unexpected)}")

    if norm_stats:
        model.action_space.load_norm_stats(norm_stats)

    model = model.to(device).eval()
    processor = SmolVLMVLAProcessor.from_pretrained(smolvlm_model or config.smolvlm_model_path)
    return model, processor


def summarize(name: str, x: np.ndarray):
    print(f"\n{name}: shape={x.shape}")
    print(f"  mean: {np.mean(x, axis=0).round(4).tolist()}")
    print(f"  std : {np.std(x, axis=0).round(4).tolist()}")
    print(f"  min : {np.min(x, axis=0).round(4).tolist()}")
    print(f"  max : {np.max(x, axis=0).round(4).tolist()}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--norm_stats", type=str, default=None)
    parser.add_argument("--dataset_path", type=str, default=None)
    parser.add_argument("--dataset_base_path", type=str, default=None)
    parser.add_argument("--task", type=str, default="CloseFridge")
    parser.add_argument("--split", type=str, default="target", choices=["pretrain", "target", "real"])
    parser.add_argument("--source", type=str, default="human")
    parser.add_argument("--num_samples", type=int, default=32)
    parser.add_argument("--num_actions", type=int, default=None)
    parser.add_argument("--sampling_steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--smolvlm_model", type=str, default="HuggingFaceTB/SmolVLM-500M-Instruct")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"

    dataset_path = args.dataset_path
    if dataset_path is None:
        if args.dataset_base_path is None:
            raise ValueError("Pass --dataset_path or --dataset_base_path.")
        dataset_path = resolve_dataset_path(args.dataset_base_path, args.task, args.split, args.source)

    model, processor = load_model(
        checkpoint=args.checkpoint,
        norm_stats=args.norm_stats,
        smolvlm_model=args.smolvlm_model,
        device=args.device,
    )

    from robocasa.utils.groot_utils.groot_dataset import LeRobotSingleDataset
    from robocasa.utils.groot_utils.schema import EmbodimentTag

    num_actions = args.num_actions or int(model.num_actions)
    dataset = LeRobotSingleDataset(
        dataset_path=dataset_path,
        modality_configs=make_robocasa_modality_configs(num_actions),
        embodiment_tag=EmbodimentTag("new_embodiment"),
        video_backend=os.environ.get("ROBOCASA_VIDEO_BACKEND", "opencv"),
        filter_key=None,
    )

    rng = random.Random(args.seed)
    sample_indices = rng.sample(range(len(dataset)), min(args.num_samples, len(dataset)))

    pred_all = []
    expert_all = []
    abs_err_first = []
    abs_err_chunk = []

    for i, dataset_index in enumerate(sample_indices):
        traj_id, base_idx = dataset.all_steps[dataset_index]
        step = dataset.get_step_data(traj_id, base_idx)
        images = [image_frame(step, key) for key in ROBOCASA_VIDEO_KEYS]
        state = concat_state(step)
        expert = concat_action(step)[:num_actions]
        prompt = get_language(step)

        image_inputs = processor.encode_image(images)
        lang = processor.encode_language([prompt])
        with torch.no_grad():
            pred = model.generate_actions(
                input_ids=lang["input_ids"].to(args.device),
                image_input=image_inputs["image_input"].to(args.device),
                image_mask=image_inputs["image_mask"].to(args.device),
                proprio=torch.tensor(state, dtype=torch.float32, device=args.device).unsqueeze(0),
                steps=args.sampling_steps,
            ).squeeze(0).detach().cpu().numpy()[:num_actions]

        pred_all.append(pred.reshape(-1, pred.shape[-1]))
        expert_all.append(expert.reshape(-1, expert.shape[-1]))
        abs_err_first.append(np.abs(pred[0] - expert[0]))
        abs_err_chunk.append(np.mean(np.abs(pred - expert), axis=0))
        print(
            f"[{i + 1}/{len(sample_indices)}] traj={traj_id} step={base_idx} "
            f"mae_first={np.mean(abs_err_first[-1]):.4f} mae_chunk={np.mean(abs_err_chunk[-1]):.4f}"
        )

    pred_all = np.concatenate(pred_all, axis=0)
    expert_all = np.concatenate(expert_all, axis=0)
    abs_err_first = np.asarray(abs_err_first)
    abs_err_chunk = np.asarray(abs_err_chunk)

    dim_names = [
        "eef_x", "eef_y", "eef_z",
        "rot_x", "rot_y", "rot_z",
        "gripper",
        "base_0", "base_1", "base_2", "base_3",
        "control_mode",
    ]

    summarize("pred actions", pred_all)
    summarize("expert actions", expert_all)
    summarize("abs error first action", abs_err_first)
    summarize("mean abs error action chunk", abs_err_chunk)

    print("\nPer-dim MAE first action:")
    for name, value in zip(dim_names, np.mean(abs_err_first, axis=0)):
        print(f"  {name:12s}: {value:.6f}")

    if model.action_space.action_norm_stats is not None:
        stats = model.action_space.action_norm_stats
        if stats.q01 is not None and stats.q99 is not None:
            q01 = stats.q01.cpu().numpy()[: pred_all.shape[-1]]
            q99 = stats.q99.cpu().numpy()[: pred_all.shape[-1]]
            below = pred_all < q01
            above = pred_all > q99
            print("\nPredicted action outside training q01/q99:")
            for name, lo, hi in zip(dim_names, np.mean(below, axis=0), np.mean(above, axis=0)):
                print(f"  {name:12s}: below={lo * 100:5.1f}% above={hi * 100:5.1f}%")


if __name__ == "__main__":
    main()
