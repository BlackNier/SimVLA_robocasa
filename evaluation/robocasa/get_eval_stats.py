#!/usr/bin/env python3
"""Aggregate RoboCasa evaluation stats written by eval_robocasa.py."""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import OrderedDict

import numpy as np
from robocasa.utils.dataset_registry import TARGET_TASKS


TASK_GROUP_MAPPING = OrderedDict(
    atomic_seen=TARGET_TASKS["atomic_seen"],
    composite_seen=TARGET_TASKS["composite_seen"],
    composite_unseen=TARGET_TASKS["composite_unseen"],
)


def _latest_stats(task_dir: str):
    if not os.path.exists(task_dir):
        return None
    runs = sorted([p for p in os.listdir(task_dir) if os.path.isdir(os.path.join(task_dir, p))])
    for run in reversed(runs):
        stats_path = os.path.join(task_dir, run, "stats.json")
        if os.path.exists(stats_path):
            with open(stats_path, "r") as f:
                return json.load(f)
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, required=True)
    parser.add_argument("--split", type=str, default="target")
    parser.add_argument(
        "--task_groups",
        nargs="+",
        default=["atomic_seen", "composite_seen", "composite_unseen"],
    )
    args = parser.parse_args()

    split_dir = os.path.join(args.dir, "evals", args.split)
    all_group_avgs = []

    for group in args.task_groups:
        values = []
        print(f"\n{group}")
        for task in TASK_GROUP_MAPPING[group]:
            stats = _latest_stats(os.path.join(split_dir, task))
            if stats is None:
                print(f"  {task}: -")
                continue
            sr = float(stats["success_rate"]) * 100.0
            values.append(sr)
            print(f"  {task}: {math.floor(sr + 0.5)}")
        avg = float(np.mean(values)) if values else float("nan")
        all_group_avgs.append(avg)
        print(f"  AVG: {avg:.1f}")

    valid = [x for x in all_group_avgs if not np.isnan(x)]
    if valid:
        print(f"\nOVERALL AVG: {np.mean(valid):.1f}")


if __name__ == "__main__":
    main()
