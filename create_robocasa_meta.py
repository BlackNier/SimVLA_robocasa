#!/usr/bin/env python3
"""Create SimVLA training metadata for RoboCasa365 LeRobot datasets."""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datasets.domain_handler.robocasa_lerobot import create_robocasa_meta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_soup", type=str, required=True)
    parser.add_argument("--dataset_base_path", type=str, default=None)
    parser.add_argument("--split", type=str, default="target", choices=["pretrain", "target", "real"])
    parser.add_argument("--source", type=str, default="human")
    parser.add_argument("--demo_fraction", type=float, default=1.0)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    meta = create_robocasa_meta(
        dataset_soup=args.dataset_soup,
        output_path=args.output,
        dataset_base_path=args.dataset_base_path,
        split=args.split,
        source=args.source,
        demo_fraction=args.demo_fraction,
    )

    missing = [item["path"] for item in meta["datalist"] if not os.path.exists(item["path"])]
    print(f"Created {args.output}")
    print(f"dataset_soup={args.dataset_soup}, num_datasets={len(meta['datalist'])}")
    if missing:
        print(f"Missing dataset paths: {len(missing)}")
        for path in missing[:20]:
            print(f"  {path}")
        if len(missing) > 20:
            print(f"  ... {len(missing) - 20} more")
        if args.strict:
            raise FileNotFoundError("Some RoboCasa dataset paths are missing.")


if __name__ == "__main__":
    main()
