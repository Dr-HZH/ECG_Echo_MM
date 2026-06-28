#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets.echo_dicom_dataset import EchoDicomDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Precompute resized Echo DICOM cache for phase2 training.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--fold", type=int, default=None, help="Optional fold filter using split file from config.")
    parser.add_argument(
        "--split",
        default=None,
        choices=["train", "inner_val", "outer_test"],
        help="Optional split filter when --fold is provided.",
    )
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    data_cfg = config["data"]
    if not data_cfg.get("cache_dir"):
        raise ValueError("Config data.cache_dir must be set for cache precomputation.")

    manifest = pd.read_csv(data_cfg["csv_path"], low_memory=False)
    if "eligible_for_training" in manifest.columns:
        manifest = manifest[manifest["eligible_for_training"] == 1].copy()

    if args.fold is not None:
        split_path = config["cross_validation"]["split_file"]
        split_df = pd.read_csv(split_path, low_memory=False)
        split_df = split_df[split_df["fold"] == args.fold].copy()
        if args.split is not None:
            split_df = split_df[split_df["split"] == args.split].copy()
        manifest = manifest.merge(
            split_df[[data_cfg["id_col"], data_cfg["patient_id_col"]]],
            on=[data_cfg["id_col"], data_cfg["patient_id_col"]],
            how="inner",
        )

    if args.limit is not None:
        manifest = manifest.iloc[: args.limit].reset_index(drop=True)

    dataset = EchoDicomDataset(
        csv_path=data_cfg["csv_path"],
        dataframe=manifest,
        echo_path_col=data_cfg["path_col"],
        label_col=data_cfg["label_col"],
        patient_id_col=data_cfg["patient_id_col"],
        split="val",
        num_frames=data_cfg["num_frames"],
        image_size=data_cfg["image_size"],
        target_channels=data_cfg.get("target_channels", 3),
        input_format=data_cfg.get("input_format", "dicom"),
        data_root=data_cfg["data_root"],
        cache_dir=data_cfg["cache_dir"],
        cache_image_size=data_cfg.get("cache_image_size", data_cfg.get("image_size")),
        crop_scale_range=None,
        brightness_jitter=0.0,
        contrast_jitter=0.0,
        translate_fraction=0.0,
        normalize_mean=None,
        normalize_std=None,
    )

    start = time.time()
    n_cached = 0
    for idx in range(len(dataset)):
        row = manifest.iloc[idx]
        abs_path = dataset._resolve_path(str(row[data_cfg["path_col"]]))
        _, meta = dataset._load_array(abs_path)
        n_cached += 1
        print(
            json.dumps(
                {
                    "event": "cache_item",
                    "index": idx,
                    "echo_id": row[data_cfg["id_col"]],
                    "path": str(abs_path),
                    "loaded_from_cache": bool(meta.get("loaded_from_cache", False)),
                    "cache_path": meta.get("cache_path"),
                }
            ),
            flush=True,
        )

    print(
        json.dumps(
            {
                "event": "cache_complete",
                "count": n_cached,
                "seconds": round(time.time() - start, 3),
                "cache_dir": str(data_cfg["cache_dir"]),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
