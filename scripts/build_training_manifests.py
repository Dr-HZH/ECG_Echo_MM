#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build training manifests from resolved CSVs.")
    parser.add_argument(
        "--multimodal_csv",
        default="resolved_csvs/301/multimodal_pairs_relative_resolved.csv",
    )
    parser.add_argument(
        "--echo_csv",
        default="resolved_csvs/301/echo_allef_relative_resolved.csv",
    )
    parser.add_argument(
        "--ecg_csv",
        default="resolved_csvs/301/ecg_hfref_relative_resolved.csv",
    )
    parser.add_argument(
        "--audit_dir",
        default="reports/data_audit_301",
    )
    parser.add_argument(
        "--echo_root",
        default="/media/data1/huangzihao/ECG_Echo_MM/301/echo_only_dcm",
    )
    parser.add_argument(
        "--ecg_root",
        default="/media/data1/huangzihao/ECG_Echo_MM/301/ecg-diagnostic-electrocardiogram-matched-subset",
    )
    parser.add_argument(
        "--output_dir",
        default="manifests/301",
    )
    parser.add_argument(
        "--main_label_col",
        default="is_ischemic_strict",
    )
    return parser.parse_args()


def load_readability_map(path: Path, path_col: str, ok_col: str) -> dict[str, int]:
    if not path.exists():
        return {}
    df = pd.read_csv(path, low_memory=False)
    if path_col not in df.columns or ok_col not in df.columns:
        return {}
    return {
        str(row[path_col]): int(row[ok_col])
        for _, row in df[[path_col, ok_col]].dropna().drop_duplicates().iterrows()
    }


def ensure_subject_id(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "subject_id" not in out.columns and "patient_id" in out.columns:
        out["subject_id"] = out["patient_id"]
    return out


def bool_or_nan(value: Any) -> Any:
    if pd.isna(value):
        return np.nan
    return int(value)


def int_flag(value: Any) -> int:
    numeric = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric):
        return 0
    return int(numeric)


def join_reasons(reasons: list[str]) -> str:
    return ";".join(reasons)


def build_multimodal_manifest(
    df: pd.DataFrame,
    main_label_col: str,
    echo_root: Path,
    ecg_root: Path,
    echo_readable: dict[str, int],
    ecg_readable: dict[str, int],
) -> pd.DataFrame:
    out = ensure_subject_id(df)
    out.insert(0, "pair_id", [f"pair_{idx:06d}" for idx in range(len(out))])
    out["echo_id"] = out.get("echo_dcm_path")
    out["ecg_id"] = out.get("ecg_record_path")
    out["pair_duplicate_count"] = (
        out.groupby(["subject_id", "echo_dcm_path", "ecg_record_path"], dropna=False)["subject_id"]
        .transform("size")
    )
    out["echo_pair_count"] = out.groupby("echo_dcm_path", dropna=False)["echo_dcm_path"].transform("size")
    out["ecg_pair_count"] = out.groupby("ecg_record_path", dropna=False)["ecg_record_path"].transform("size")
    out["echo_dcm_abs_path"] = out["echo_dcm_path"].apply(
        lambda x: str(echo_root / x) if pd.notna(x) else np.nan
    )
    out["ecg_record_abs_path"] = out["ecg_record_path"].apply(
        lambda x: str(ecg_root / x) if pd.notna(x) else np.nan
    )
    out["readable_echo"] = out["echo_dcm_path"].map(echo_readable)
    out["readable_ecg"] = out["ecg_record_path"].map(ecg_readable)

    eligible = []
    reasons = []
    for _, row in out.iterrows():
        row_reasons: list[str] = []
        if str(row.get("mapping_status", "")) != "ok":
            row_reasons.append(f"mapping_status={row.get('mapping_status')}")
        if int_flag(row.get("echo_dcm_exists")) != 1:
            row_reasons.append("missing_echo_dcm")
        if int_flag(row.get("ecg_record_exists")) != 1:
            row_reasons.append("missing_ecg_record")
        if pd.isna(row.get("subject_id")):
            row_reasons.append("missing_subject_id")
        if main_label_col not in out.columns:
            row_reasons.append(f"missing_main_label_col:{main_label_col}")
        elif pd.isna(row.get(main_label_col)):
            row_reasons.append("missing_main_label_value")
        readable_echo = row.get("readable_echo")
        readable_ecg = row.get("readable_ecg")
        if pd.notna(readable_echo) and int(readable_echo) != 1:
            row_reasons.append("unreadable_echo")
        if pd.notna(readable_ecg) and int(readable_ecg) != 1:
            row_reasons.append("unreadable_ecg")
        eligible.append(int(len(row_reasons) == 0))
        reasons.append(join_reasons(row_reasons))

    out["eligible_for_training"] = eligible
    out["exclusion_reason"] = reasons
    return out


def build_echo_manifest(
    df: pd.DataFrame,
    echo_root: Path,
    echo_readable: dict[str, int],
) -> pd.DataFrame:
    out = ensure_subject_id(df)
    out = out[pd.to_numeric(out.get("echo_dcm_exists"), errors="coerce") == 1].copy()
    out.insert(0, "echo_id", [f"echo_{idx:06d}" for idx in range(len(out))])
    out["echo_dcm_abs_path"] = out["echo_dcm_path"].apply(
        lambda x: str(echo_root / x) if pd.notna(x) else np.nan
    )
    out["readable_echo"] = out["echo_dcm_path"].map(echo_readable)
    out["eligible_for_training"] = [
        int(pd.notna(path) and (pd.isna(readable) or int(readable) == 1))
        for path, readable in zip(out["echo_dcm_path"], out["readable_echo"])
    ]
    out["exclusion_reason"] = [
        "" if eligible else "unreadable_echo"
        for eligible in out["eligible_for_training"]
    ]
    return out


def build_ecg_manifest(
    df: pd.DataFrame,
    ecg_root: Path,
    ecg_readable: dict[str, int],
) -> pd.DataFrame:
    out = ensure_subject_id(df)
    out = out[pd.to_numeric(out.get("ecg_record_exists"), errors="coerce") == 1].copy()
    out.insert(0, "ecg_id", [f"ecg_{idx:06d}" for idx in range(len(out))])
    out["ecg_record_abs_path"] = out["ecg_record_path"].apply(
        lambda x: str(ecg_root / x) if pd.notna(x) else np.nan
    )
    out["readable_ecg"] = out["ecg_record_path"].map(ecg_readable)
    out["eligible_for_training"] = [
        int(pd.notna(path) and (pd.isna(readable) or int(readable) == 1))
        for path, readable in zip(out["ecg_record_path"], out["readable_ecg"])
    ]
    out["exclusion_reason"] = [
        "" if eligible else "unreadable_ecg"
        for eligible in out["eligible_for_training"]
    ]
    return out


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    audit_dir = Path(args.audit_dir)

    multimodal_df = pd.read_csv(args.multimodal_csv, low_memory=False)
    echo_df = pd.read_csv(args.echo_csv, low_memory=False)
    ecg_df = pd.read_csv(args.ecg_csv, low_memory=False)

    echo_readable = load_readability_map(audit_dir / "echo_file_stats.csv", "relative_path", "pixel_array_ok")
    ecg_readable = load_readability_map(audit_dir / "ecg_file_stats.csv", "relative_path", "rdrecord_ok")

    multimodal_manifest = build_multimodal_manifest(
        multimodal_df,
        main_label_col=args.main_label_col,
        echo_root=Path(args.echo_root),
        ecg_root=Path(args.ecg_root),
        echo_readable=echo_readable,
        ecg_readable=ecg_readable,
    )
    echo_manifest = build_echo_manifest(
        echo_df,
        echo_root=Path(args.echo_root),
        echo_readable=echo_readable,
    )
    ecg_manifest = build_ecg_manifest(
        ecg_df,
        ecg_root=Path(args.ecg_root),
        ecg_readable=ecg_readable,
    )

    multimodal_manifest.to_csv(output_dir / "multimodal_pairs_manifest.csv", index=False)
    echo_manifest.to_csv(output_dir / "echo_available_manifest.csv", index=False)
    ecg_manifest.to_csv(output_dir / "ecg_available_manifest.csv", index=False)

    summary = {
        "main_label_col": args.main_label_col,
        "multimodal_rows": int(len(multimodal_manifest)),
        "multimodal_eligible": int(multimodal_manifest["eligible_for_training"].sum()),
        "echo_rows": int(len(echo_manifest)),
        "echo_eligible": int(echo_manifest["eligible_for_training"].sum()),
        "ecg_rows": int(len(ecg_manifest)),
        "ecg_eligible": int(ecg_manifest["eligible_for_training"].sum()),
    }
    (output_dir / "manifest_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
