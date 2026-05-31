#!/usr/bin/env python3
"""Evaluate a SimVLA WebSocket policy server on RoboCasa365 target/pretrain splits."""

from __future__ import annotations

import argparse
import asyncio
import collections
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import gymnasium as gym
import imageio
import numpy as np
import tqdm

try:
    import msgpack
    import msgpack_numpy

    HAS_MSGPACK = True
except ImportError:
    HAS_MSGPACK = False

import websockets

import robocasa
from robocasa.utils.dataset_registry import TASK_SET_REGISTRY
from robocasa.utils.dataset_registry_utils import get_task_horizon
from robocasa.utils.env_utils import convert_action


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("eval_robocasa")


class WebSocketPolicyClient:
    def __init__(self, host: str, port: int):
        self.uri = f"ws://{host}:{port}"
        self.websocket = None

    async def connect(self):
        self.websocket = await websockets.connect(self.uri, max_size=None)
        await self.websocket.recv()

    async def infer(self, payload: dict[str, Any]) -> dict[str, Any]:
        assert self.websocket is not None
        if HAS_MSGPACK:
            await self.websocket.send(msgpack_numpy.packb(payload, use_bin_type=True))
            response = await self.websocket.recv()
            return msgpack.unpackb(response, raw=False)

        await self.websocket.send(json.dumps(_jsonify(payload)))
        return json.loads(await self.websocket.recv())

    async def close(self):
        if self.websocket is not None:
            await self.websocket.close()


def _jsonify(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, dict):
        return {k: _jsonify(v) for k, v in x.items()}
    return x


def _state_from_obs(obs: dict[str, Any]) -> np.ndarray:
    return np.concatenate(
        [
            obs["state.end_effector_position_relative"],
            obs["state.end_effector_rotation_relative"],
            obs["state.base_position"],
            obs["state.base_rotation"],
            obs["state.gripper_qpos"],
        ],
        axis=0,
    ).astype(np.float32)


async def _eval_env(
    env_name: str,
    split: str,
    log_dir: str,
    num_trials: int,
    replan_steps: int,
    host: str,
    port: int,
    seed: int,
    max_steps: int | None,
    save_video: bool,
):
    horizon = get_task_horizon(env_name)
    if max_steps is not None:
        horizon = min(horizon, max_steps)

    log_path = Path(log_dir) / "evals" / split / env_name 
    log_path.mkdir(parents=True, exist_ok=True)

    env = gym.make(f"robocasa/{env_name}", split=split, seed=seed)
    client = WebSocketPolicyClient(host, port)
    await client.connect()

    successes = 0
    for episode_idx in tqdm.tqdm(range(num_trials), desc=env_name):
        obs, info = env.reset()
        task_lang = obs["annotation.human.task_description"]
        action_plan = collections.deque()
        replay_images = []
        done = False

        for t in range(horizon):
            if not action_plan:
                payload = {
                    "observation/image": np.ascontiguousarray(obs["video.robot0_agentview_left"]),
                    "observation/wrist_image": np.ascontiguousarray(obs["video.robot0_eye_in_hand"]),
                    "observation/right_image": np.ascontiguousarray(obs["video.robot0_agentview_right"]),
                    "observation/state": _state_from_obs(obs),
                    "prompt": task_lang,
                }
                action_chunk = np.asarray((await client.infer(payload))["actions"], dtype=np.float32)
                if len(action_chunk) < replan_steps:
                    raise ValueError(
                        f"Policy returned {len(action_chunk)} actions, but replan_steps={replan_steps}."
                    )
                action_plan.extend(action_chunk[:replan_steps])

            action = convert_action(np.asarray(action_plan.popleft(), dtype=np.float32))
            obs, reward, terminated, truncated, info = env.step(action)
            done = bool(info["success"])

            if save_video and (t % 2 == 0 or done or t == horizon - 1):
                replay_images.append(np.ascontiguousarray(env.render()))

            if done:
                successes += 1
                break

        if save_video:
            suffix = "success" if done else "failure"
            imageio.mimwrite(log_path / f"rollout_{episode_idx}_{suffix}.mp4", replay_images, fps=20)

    stats = {
        "task": env_name,
        "split": split,
        "num_episodes": num_trials,
        "success_rate": successes / float(num_trials),
    }
    with open(log_path / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    await client.close()
    env.close()
    print(f"{env_name}: {stats['success_rate'] * 100:.1f}% ({successes}/{num_trials})")
    return stats


async def _main_async(args):
    task_names = []
    for task_set in args.task_set:
        if task_set in TASK_SET_REGISTRY:
            task_names.extend(TASK_SET_REGISTRY[task_set])
        else:
            task_names.append(task_set)
    task_names = list(dict.fromkeys(task_names))

    all_stats = []
    for env_name in task_names:
        all_stats.append(
            await _eval_env(
                env_name=env_name,
                split=args.split,
                log_dir=args.log_dir,
                num_trials=args.num_trials,
                replan_steps=args.replan_steps,
                host=args.host,
                port=args.port,
                seed=args.seed,
                max_steps=args.max_steps,
                save_video=not args.no_video,
            )
        )

    print("Summary")
    for stat in all_stats:
        print(f"  {stat['task']}: {stat['success_rate'] * 100:.1f}%")
    print(f"  average: {np.mean([s['success_rate'] for s in all_stats]) * 100:.1f}%")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--split", type=str, default="target", choices=["pretrain", "target"])
    parser.add_argument("--task_set", nargs="+", default=["atomic_seen"])
    parser.add_argument("--num_trials", type=int, default=50)
    parser.add_argument("--replan_steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--log_dir", type=str, required=True)
    parser.add_argument("--no_video", action="store_true")
    args = parser.parse_args()
    asyncio.run(_main_async(args))


if __name__ == "__main__":
    main()
