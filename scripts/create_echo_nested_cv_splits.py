#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.cv_split import build_nested_stratified_group_splits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create nested patient-level CV splits for Echo phase2.")
    parser.add_argument("--manifest", default="manifests/301/echo_hfref_manifest.csv")
    parser.add_argument("--id_col", default="echo_id")
    parser.add_argument("--patient_col", default="subject_id")
    parser.add_argument("--label_col", default="main_label")
    parser.add_argument("--output_csv", default="splits/301_echo_nested_5fold.csv")
    parser.add_argument("--summary_json", default="splits/301_echo_nested_5fold_summary.json")
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--inner_val_ratio", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = pd.read_csv(args.manifest, low_memory=False)
    if "eligible_for_training" in manifest.columns:
        manifest = manifest[manifest["eligible_for_training"] == 1].copy()
    artifacts = build_nested_stratified_group_splits(
        manifest,
        id_col=args.id_col,
        patient_col=args.patient_col,
        label_col=args.label_col,
        n_splits=args.n_splits,
        inner_val_ratio=args.inner_val_ratio,
        seed=args.seed,
    )
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    artifacts.assignments.to_csv(output_csv, index=False)
    summary_json = Path(args.summary_json)
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(artifacts.summary, indent=2), encoding="utf-8")
    print(json.dumps(artifacts.summary, indent=2))


if __name__ == "__main__":
    main()
