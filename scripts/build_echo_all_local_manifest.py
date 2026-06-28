#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(description="Freeze the locally mapped Echo DICOM cohort.")
    p.add_argument("--source", default="resolved_csvs/301/echo_allef_relative_resolved.csv")
    p.add_argument("--data_root", default="301/echo_only_dcm")
    p.add_argument("--output", default="manifests/echo_all_local_manifest.csv")
    p.add_argument("--summary", default="manifests/echo_all_local_manifest_summary.json")
    return p.parse_args()


def main():
    args = parse_args()
    df = pd.read_csv(args.source, low_memory=False)
    required = ["subject_id", "echo_dcm_path", "echo_is_ischemic", "ef_hfref"]
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise KeyError(f"Missing columns: {missing}")

    df["echo_is_ischemic"] = pd.to_numeric(df["echo_is_ischemic"], errors="coerce")
    df["ef_hfref"] = pd.to_numeric(df["ef_hfref"], errors="coerce")
    df["is_hfref_subgroup"] = df["ef_hfref"].eq(1).astype(int)
    df["dcm_exists_runtime"] = df["echo_dcm_path"].map(
        lambda value: int(pd.notna(value) and (Path(args.data_root) / str(value)).is_file())
    )
    df["duplicate_count"] = df.groupby("echo_dcm_path", dropna=False)["echo_dcm_path"].transform("size")

    candidates = df[
        df["subject_id"].notna()
        & df["echo_dcm_path"].notna()
        & df["echo_is_ischemic"].notna()
        & df["dcm_exists_runtime"].eq(1)
    ].copy()
    conflicts = candidates.groupby("echo_dcm_path")["echo_is_ischemic"].nunique().gt(1)
    conflict_paths = set(conflicts[conflicts].index)
    if conflict_paths:
        raise RuntimeError(f"Duplicate paths with conflicting labels: {len(conflict_paths)}")
    candidates = candidates.drop_duplicates("echo_dcm_path").reset_index(drop=True)
    candidates.insert(0, "sample_id", [f"echo_local_{i:06d}" for i in range(len(candidates))])
    candidates["readable"] = np.nan
    candidates["eligible_for_training"] = 1
    candidates["exclusion_reason"] = "pending_full_decode_audit"
    candidates["main_label"] = candidates["echo_is_ischemic"].astype(int)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    candidates.to_csv(output, index=False)
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    summary = {
        "source_rows": int(len(df)),
        "mapped_unique_dicom": int(len(candidates)),
        "unique_patients": int(candidates.subject_id.nunique()),
        "positive": int(candidates.echo_is_ischemic.sum()),
        "positive_rate": float(candidates.echo_is_ischemic.mean()),
        "hfref_samples": int(candidates.is_hfref_subgroup.sum()),
        "hfref_positive": int(candidates.loc[candidates.is_hfref_subgroup.eq(1), "echo_is_ischemic"].sum()),
        "manifest_sha256": digest,
        "status": "pending_full_decode_audit",
    }
    Path(args.summary).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
