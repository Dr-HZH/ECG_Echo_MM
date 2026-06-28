#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the final Phase 3 HFrEF Echo manifest from resolved Echo CSV.")
    parser.add_argument("--resolved_csv", default="resolved_csvs/301/echo_allef_relative_resolved.csv")
    parser.add_argument("--dcm_root", default="301/echo_only_dcm")
    parser.add_argument("--output_csv", default="manifests/echo_hfref_final_dcm_manifest.csv")
    parser.add_argument("--summary_json", default="manifests/echo_hfref_final_dcm_manifest_summary.json")
    parser.add_argument("--path_col", default="echo_dcm_path")
    parser.add_argument("--patient_col", default="subject_id")
    parser.add_argument("--study_col", default="echo_study_id")
    parser.add_argument("--hfref_col", default="ef_hfref")
    parser.add_argument("--tier_col", default="echo_master_ischemia_tier")
    parser.add_argument("--aux_cols", nargs="*", default=[
        "echo_rwma_feature_flag",
        "echo_scar_feature_flag",
        "echo_severe_cad_flag",
        "ef",
    ])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    resolved_path = Path(args.resolved_csv)
    if not resolved_path.exists():
        raise FileNotFoundError(resolved_path)

    df = pd.read_csv(resolved_path, low_memory=False)
    required = [args.path_col, args.patient_col, args.study_col, args.hfref_col, args.tier_col]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    dcm_root = Path(args.dcm_root)
    working = df.copy()
    working[args.hfref_col] = pd.to_numeric(working[args.hfref_col], errors="coerce")
    working = working[working[args.hfref_col] == 1].copy()
    working = working[working[args.path_col].notna()].copy()
    working = working.drop_duplicates(subset=[args.path_col]).reset_index(drop=True)

    tier = working[args.tier_col].fillna("").astype(str).str.strip().str.upper()
    working["final_ischemia_label"] = tier.isin(["A", "B"]).astype(int)
    working["dcm_exists_runtime"] = working[args.path_col].map(lambda value: int((dcm_root / str(value)).exists()))
    working["eligible_for_training"] = (
        (working["dcm_exists_runtime"] == 1)
        & working[args.patient_col].notna()
        & working["final_ischemia_label"].notna()
    ).astype(int)

    working.insert(0, "sample_id", [f"echo_hfref_{idx:06d}" for idx in range(len(working))])
    working["main_label"] = working["final_ischemia_label"]

    keep_cols = [
        "sample_id",
        args.patient_col,
        args.study_col,
        args.path_col,
        args.hfref_col,
        args.tier_col,
        "final_ischemia_label",
        "main_label",
        "dcm_exists_runtime",
        "eligible_for_training",
    ]
    keep_cols.extend([col for col in args.aux_cols if col in working.columns])
    keep_cols = list(dict.fromkeys(keep_cols))
    output = working[keep_cols].copy()

    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output_csv, index=False)

    eligible = output[output["eligible_for_training"] == 1]
    summary = {
        "source_csv": str(resolved_path),
        "dcm_root": str(dcm_root),
        "rows_hfref_unique_dcm": int(len(output)),
        "eligible_for_training": int(output["eligible_for_training"].sum()),
        "missing_dcm": int((output["dcm_exists_runtime"] == 0).sum()),
        "positive_tier_ab": int(eligible["final_ischemia_label"].sum()),
        "negative_tier_not_ab": int((eligible["final_ischemia_label"] == 0).sum()),
        "positive_rate": float(eligible["final_ischemia_label"].mean()) if len(eligible) else None,
        "unique_patients": int(eligible[args.patient_col].nunique()),
        "aux_cols_present": [col for col in args.aux_cols if col in output.columns],
    }
    Path(args.summary_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary_json).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))

    if summary["eligible_for_training"] != 2556:
        raise RuntimeError(f"Expected 2556 eligible HFrEF Echo rows, got {summary['eligible_for_training']}.")
    if summary["positive_tier_ab"] != 1073:
        raise RuntimeError(f"Expected 1073 tier A/B positives, got {summary['positive_tier_ab']}.")
    if summary["missing_dcm"] != 0:
        raise RuntimeError("Some HFrEF DICOM files are still missing.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(json.dumps({"error": str(exc)}), file=sys.stderr)
        raise
