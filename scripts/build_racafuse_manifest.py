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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build 30-day paired HDF5 manifest and patient-level CV splits.")
    parser.add_argument("--pairs_csv", default="resolved_csvs/301/multimodal_pairs_relative_resolved.csv")
    parser.add_argument("--echo_manifest", default="manifests/echo_all_local_manifest.csv")
    parser.add_argument("--ecg_h5_manifest", default="manifests/301/ecg_h5_manifest.csv")
    parser.add_argument("--echo_h5_dir", default="data_h5/echo_all_local_t32_224_lzf")
    parser.add_argument("--output_manifest", default="manifests/racafuse_pairs_30d_h5_manifest.csv")
    parser.add_argument("--output_split", default="splits/racafuse_pairs_30d_5fold.csv")
    parser.add_argument("--summary_json", default="reports/racafuse_pairs_30d_manifest_summary.json")
    parser.add_argument("--max_days", type=float, default=30.0)
    parser.add_argument("--label_col", default="echo_strict_is_ischemic")
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--inner_val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def build_echo_h5_map(echo_h5_dir: Path) -> dict[str, tuple[str, str]]:
    try:
        import h5py  # type: ignore
    except Exception as exc:
        raise RuntimeError("h5py is required to scan echo HDF5 shards.") from exc

    mapping: dict[str, tuple[str, str]] = {}
    for path in sorted(echo_h5_dir.glob("part_*/shard_*.h5")):
        try:
            with h5py.File(path, "r") as handle:
                if "videos" not in handle:
                    continue
                for key in handle["videos"].keys():
                    mapping[str(key)] = (str(path), f"/videos/{key}")
        except Exception as exc:
            print(f"Warning: skipping unreadable echo HDF5 shard {path}: {exc}", file=sys.stderr)
    return mapping


def main() -> None:
    args = parse_args()
    from utils.cv_split import build_nested_stratified_group_splits

    pairs = pd.read_csv(args.pairs_csv, low_memory=False)
    pairs = pairs[pd.to_numeric(pairs["abs_time_diff_days"], errors="coerce") <= args.max_days].copy()
    pairs[args.label_col] = pd.to_numeric(pairs[args.label_col], errors="coerce")
    pairs = pairs[pairs[args.label_col].isin([0, 1])].copy()
    pairs["label"] = pairs[args.label_col].astype(int)

    echo_manifest = pd.read_csv(args.echo_manifest, low_memory=False)
    if {"h5_path", "h5_key"}.issubset(echo_manifest.columns):
        id_col = "sample_id" if "sample_id" in echo_manifest.columns else "echo_id"
        echo_lookup = echo_manifest[["echo_dcm_path", id_col, "h5_path", "h5_key"]].dropna().drop_duplicates("echo_dcm_path")
        echo_lookup = echo_lookup.rename(
            columns={id_col: "echo_sample_id", "h5_path": "echo_h5_path", "h5_key": "echo_h5_key"}
        )
        echo_h5_key_count = int(echo_lookup["echo_sample_id"].nunique())
    else:
        echo_h5_map = build_echo_h5_map(Path(args.echo_h5_dir))
        echo_lookup = echo_manifest[["echo_dcm_path", "sample_id"]].dropna().drop_duplicates("echo_dcm_path")
        echo_lookup = echo_lookup.rename(columns={"sample_id": "echo_sample_id"})
        echo_lookup["echo_h5_path"] = echo_lookup["echo_sample_id"].map(lambda x: echo_h5_map.get(str(x), ("", ""))[0])
        echo_lookup["echo_h5_key"] = echo_lookup["echo_sample_id"].map(lambda x: echo_h5_map.get(str(x), ("", ""))[1])
        echo_lookup = echo_lookup[(echo_lookup["echo_h5_path"] != "") & (echo_lookup["echo_h5_key"] != "")]
        echo_h5_key_count = int(len(echo_h5_map))

    ecg_manifest = pd.read_csv(args.ecg_h5_manifest, low_memory=False)
    ecg_lookup = ecg_manifest[["ecg_record_path", "h5_path", "h5_key"]].dropna().drop_duplicates("ecg_record_path")
    ecg_lookup = ecg_lookup.rename(columns={"h5_path": "ecg_h5_path", "h5_key": "ecg_h5_key"})

    merged = pairs.merge(echo_lookup, on="echo_dcm_path", how="inner")
    merged = merged.merge(ecg_lookup, on="ecg_record_path", how="inner")
    merged = merged.reset_index(drop=True)
    merged.insert(0, "sample_id", [f"racafuse_{i:06d}" for i in range(len(merged))])
    merged["time_delta_days"] = pd.to_numeric(merged["abs_time_diff_days"], errors="coerce").fillna(0.0)

    columns = [
        "sample_id",
        "patient_id",
        "label",
        "time_delta_days",
        "echo_dcm_path",
        "ecg_record_path",
        "echo_h5_path",
        "echo_h5_key",
        "ecg_h5_path",
        "ecg_h5_key",
        "echo_study_id",
        "ecg_study_id",
        "echo_master_ischemia_tier",
        "strict_pair_label",
        "rwma_feature_flag",
        "scar_feature_flag",
        "severe_cad_flag",
    ]
    out = merged[[col for col in columns if col in merged.columns]].copy()

    Path(args.output_manifest).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_split).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary_json).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output_manifest, index=False)

    cv = build_nested_stratified_group_splits(
        out,
        id_col="sample_id",
        patient_col="patient_id",
        label_col="label",
        n_splits=args.n_splits,
        inner_val_ratio=args.inner_val_ratio,
        seed=args.seed,
    )
    cv.assignments.to_csv(args.output_split, index=False)
    summary = {
        "pairs_30d_labeled": int(len(pairs)),
        "echo_h5_keys": echo_h5_key_count,
        "matched_rows": int(len(out)),
        "unique_patients": int(out["patient_id"].nunique()),
        "positive": int((out["label"] == 1).sum()),
        "negative": int((out["label"] == 0).sum()),
        "positive_rate": float(out["label"].mean()),
        "cv": cv.summary,
    }
    Path(args.summary_json).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
