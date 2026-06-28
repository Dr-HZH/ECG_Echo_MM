#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit delivered Echo DICOM coverage for the final HFrEF cohort.")
    parser.add_argument("--csv", default="resolved_csvs/301/echo_allef_relative_resolved.csv")
    parser.add_argument("--dcm_root", default="301/echo_only_dcm")
    parser.add_argument("--output_dir", default="reports/phase2_echo/delivery_audit")
    args = parser.parse_args()

    df = pd.read_csv(args.csv, low_memory=False)
    df["ef_hfref_numeric"] = pd.to_numeric(df["ef_hfref"], errors="coerce")
    df["final_ischemia_label"] = df["echo_master_ischemia_tier"].isin(["A", "B"]).astype(int)
    root = Path(args.dcm_root)
    df["dcm_exists_runtime"] = df["echo_dcm_path"].map(
        lambda value: int(pd.notna(value) and (root / str(value)).is_file())
    )

    hfref = df[df["ef_hfref_numeric"].eq(1)].drop_duplicates("echo_dcm_path").copy()
    delivered = hfref[hfref["dcm_exists_runtime"].eq(1)].copy()
    missing = hfref[hfref["dcm_exists_runtime"].eq(0)].copy()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    columns = ["echo_dcm_path", "subject_id", "echo_study_id", "echo_master_ischemia_tier", "final_ischemia_label"]
    hfref[columns + ["dcm_exists_runtime"]].to_csv(output / "hfref_delivery_status.csv", index=False)
    missing["echo_dcm_path"].to_csv(output / "missing_hfref_dcm_paths.txt", index=False, header=False)

    local_files = list(root.rglob("*.dcm"))
    summary = {
        "echo_csv_rows": int(len(df)),
        "echo_csv_unique_expected_dcm": int(df["echo_dcm_path"].nunique()),
        "local_dcm_files": int(len(local_files)),
        "hfref_expected": int(len(hfref)),
        "hfref_expected_positive_tier_ab": int(hfref["final_ischemia_label"].sum()),
        "hfref_delivered": int(len(delivered)),
        "hfref_delivered_positive_tier_ab": int(delivered["final_ischemia_label"].sum()),
        "hfref_missing": int(len(missing)),
        "hfref_missing_positive_tier_ab": int(missing["final_ischemia_label"].sum()),
        "delivery_fraction": float(len(delivered) / len(hfref)),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output / "README.md").write_text(
        "# Echo Delivery Audit\n\n"
        "The delivered archive is a 5,181-file paired-data subset, not the complete Echo-only cohort.\n\n"
        f"- Final HFrEF clips expected: {len(hfref)}\n"
        f"- Delivered exact clips: {len(delivered)}\n"
        f"- Missing clips to copy: {len(missing)}\n"
        f"- Expected tier A/B positives: {int(hfref.final_ischemia_label.sum())}\n"
        f"- Missing tier A/B positives: {int(missing.final_ischemia_label.sum())}\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
