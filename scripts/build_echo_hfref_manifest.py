#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build phase2 Echo HFrEF manifest.")
    parser.add_argument(
        "--echo_csv",
        default="resolved_csvs/301/echo_allef_relative_resolved.csv",
    )
    parser.add_argument(
        "--audit_dir",
        default="reports/data_audit_301",
    )
    parser.add_argument(
        "--echo_root",
        default="301/echo_only_dcm",
    )
    parser.add_argument(
        "--output_csv",
        default="manifests/301/echo_hfref_manifest.csv",
    )
    parser.add_argument(
        "--summary_json",
        default="manifests/301/echo_hfref_manifest_summary.json",
    )
    parser.add_argument(
        "--decode_policy",
        choices=["allow_unchecked", "require_readable"],
        default="allow_unchecked",
        help=(
            "allow_unchecked: keep files with sampled audit status `unchecked`; "
            "require_readable: only keep files explicitly validated as readable."
        ),
    )
    return parser.parse_args()


def load_readability_map(path: Path) -> dict[str, int]:
    if not path.exists():
        return {}
    df = pd.read_csv(path, low_memory=False)
    if "relative_path" not in df.columns or "pixel_array_ok" not in df.columns:
        return {}
    return {
        str(row["relative_path"]): int(row["pixel_array_ok"])
        for _, row in df[["relative_path", "pixel_array_ok"]].dropna().drop_duplicates().iterrows()
    }


def main() -> None:
    args = parse_args()
    echo_df = pd.read_csv(args.echo_csv, low_memory=False)
    read_map = load_readability_map(Path(args.audit_dir) / "echo_file_stats.csv")

    out = echo_df.copy()
    out["main_label"] = pd.to_numeric(out["is_ischemic"], errors="coerce")
    out["ef_hfref"] = pd.to_numeric(out["ef_hfref"], errors="coerce")
    out["echo_dcm_exists"] = pd.to_numeric(out["echo_dcm_exists"], errors="coerce")
    out["readable_echo"] = out["echo_dcm_path"].map(read_map)
    out["decode_status"] = np.where(
        out["readable_echo"].isna(),
        "unchecked",
        np.where(out["readable_echo"] == 1, "readable", "unreadable"),
    )
    out["echo_dcm_abs_path"] = out["echo_dcm_path"].apply(
        lambda x: str(Path(args.echo_root) / x) if pd.notna(x) else np.nan
    )
    out["echo_id"] = [f"echo_{idx:06d}" for idx in range(len(out))]

    filtered = out[
        (out["ef_hfref"] == 1)
        & (out["echo_dcm_exists"] == 1)
        & out["main_label"].notna()
        & out["subject_id"].notna()
    ].copy()

    before_dedup = len(filtered)
    filtered = filtered.drop_duplicates(subset=["echo_dcm_path"]).reset_index(drop=True)
    filtered["echo_id"] = [f"echo_hfref_{idx:06d}" for idx in range(len(filtered))]
    if args.decode_policy == "require_readable":
        filtered["eligible_for_training"] = np.where(filtered["decode_status"] == "readable", 1, 0)
        filtered["exclusion_reason"] = np.where(filtered["eligible_for_training"] == 1, "", "decode_not_verified")
    else:
        filtered["eligible_for_training"] = np.where(filtered["decode_status"] == "unreadable", 0, 1)
        filtered["exclusion_reason"] = np.where(filtered["eligible_for_training"] == 1, "", "unreadable_echo")

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    filtered.to_csv(output_csv, index=False)

    summary = {
        "source_rows": int(len(out)),
        "filtered_rows_before_dedup": int(before_dedup),
        "filtered_rows_after_dedup": int(len(filtered)),
        "positive_rows_after_dedup": int(filtered["main_label"].sum()),
        "positive_rate_after_dedup": float(filtered["main_label"].mean()) if len(filtered) else None,
        "unique_subjects_after_dedup": int(filtered["subject_id"].nunique(dropna=True)),
        "decode_status_distribution": {
            str(k): int(v) for k, v in filtered["decode_status"].value_counts(dropna=False).items()
        },
        "decode_policy": args.decode_policy,
        "eligible_rows": int(filtered["eligible_for_training"].sum()),
        "deduplicated_rows_removed": int(before_dedup - len(filtered)),
    }
    summary_path = Path(args.summary_json)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
