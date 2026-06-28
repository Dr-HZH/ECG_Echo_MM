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

from utils.group_split import create_patient_splits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create patient-level train/val/test splits.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--patient_col", required=True)
    parser.add_argument("--label_col", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_ratio", type=float, default=0.70)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--test_ratio", type=float, default=0.15)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = pd.read_csv(args.manifest, low_memory=False)
    if args.label_col not in manifest.columns:
        raise SystemExit(
            f"Blocker: requested label column `{args.label_col}` is absent in {args.manifest}. "
            "Do not substitute another label implicitly."
        )
    if args.patient_col not in manifest.columns:
        raise SystemExit(f"Missing patient column `{args.patient_col}` in {args.manifest}.")

    trainable = manifest.copy()
    if "eligible_for_training" in trainable.columns:
        trainable = trainable[trainable["eligible_for_training"] == 1].copy()
    trainable[args.label_col] = pd.to_numeric(trainable[args.label_col], errors="coerce")
    trainable = trainable[trainable[args.patient_col].notna()].copy()
    if trainable.empty:
        raise SystemExit("No eligible rows remain for patient split generation.")

    artifacts = create_patient_splits(
        trainable,
        patient_col=args.patient_col,
        label_col=args.label_col,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    artifacts.patient_splits.to_csv(output_path, index=False)
    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(artifacts.summary, indent=2), encoding="utf-8")
    conflicts_path = output_path.with_suffix(".patient_conflicts.csv")
    artifacts.patient_table[artifacts.patient_table["has_conflict"]].to_csv(conflicts_path, index=False)
    print(json.dumps(artifacts.summary, indent=2))
    print(f"split_csv={output_path}")


if __name__ == "__main__":
    main()
