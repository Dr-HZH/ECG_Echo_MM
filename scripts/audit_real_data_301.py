#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.dicom_io import inspect_dicom_file
from utils.wfdb_io import inspect_wfdb_record


MAIN_LABEL_CANDIDATES = [
    "is_ischemic_strict",
    "is_ischemic",
    "strict_is_ischemic",
    "label",
]


def get_patient_col(df: pd.DataFrame) -> str | None:
    for candidate in ("subject_id", "patient_id"):
        if candidate in df.columns:
            return candidate
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit resolved CSVs and readable files for dataset 301.")
    parser.add_argument("--multimodal_csv", required=True)
    parser.add_argument("--echo_csv", required=True)
    parser.add_argument("--ecg_csv", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--echo_root",
        default="/media/data1/huangzihao/ECG_Echo_MM/301/echo_only_dcm",
    )
    parser.add_argument(
        "--ecg_root",
        default="/media/data1/huangzihao/ECG_Echo_MM/301/ecg-diagnostic-electrocardiogram-matched-subset",
    )
    parser.add_argument("--mode", choices=["sample", "full"], default="sample")
    parser.add_argument("--sample_size", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def infer_value_type(series: pd.Series) -> str:
    nonnull = series.dropna()
    if nonnull.empty:
        return "all_missing"
    if pd.api.types.is_bool_dtype(nonnull):
        return "bool"
    if pd.api.types.is_numeric_dtype(nonnull):
        values = set(pd.Series(nonnull).astype(float).unique().tolist())
        if values.issubset({0.0, 1.0}):
            return "binary_numeric"
        return "numeric"
    if pd.api.types.is_datetime64_any_dtype(nonnull):
        return "datetime"
    return "string_or_mixed"


def build_label_statistics(df: pd.DataFrame, csv_name: str) -> pd.DataFrame:
    rows = []
    for column in df.columns:
        series = df[column]
        nonnull = series.dropna()
        kind = infer_value_type(series)
        positive_rate = None
        if kind == "binary_numeric" and not nonnull.empty:
            positive_rate = float(pd.Series(nonnull).astype(float).mean())
        rows.append(
            {
                "csv_name": csv_name,
                "column": column,
                "missing_rate": float(series.isna().mean()),
                "nonnull_count": int(nonnull.shape[0]),
                "unique_nonnull": int(nonnull.nunique(dropna=True)),
                "inferred_type": kind,
                "is_main_label_candidate": int(column in MAIN_LABEL_CANDIDATES),
                "binary_positive_rate": positive_rate,
            }
        )
    return pd.DataFrame(rows)


def summarize_table(
    df: pd.DataFrame,
    name: str,
    echo_col: str | None = None,
    ecg_col: str | None = None,
) -> dict[str, Any]:
    patient_col = get_patient_col(df)
    summary: dict[str, Any] = {
        "name": name,
        "rows": int(len(df)),
        "patient_id_column": patient_col,
        "unique_subjects": int(df[patient_col].nunique(dropna=True)) if patient_col is not None else None,
        "duplicated_rows": int(df.duplicated().sum()),
    }
    if echo_col and echo_col in df.columns:
        summary["unique_echo_paths"] = int(df[echo_col].nunique(dropna=True))
        echo_counts = df[echo_col].value_counts(dropna=True)
        summary["duplicated_echo_unique"] = int((echo_counts > 1).sum())
    if ecg_col and ecg_col in df.columns:
        summary["unique_ecg_records"] = int(df[ecg_col].nunique(dropna=True))
        ecg_counts = df[ecg_col].value_counts(dropna=True)
        summary["duplicated_ecg_unique"] = int((ecg_counts > 1).sum())
    if "mapping_status" in df.columns:
        summary["mapping_status"] = {str(k): int(v) for k, v in df["mapping_status"].value_counts(dropna=False).items()}
    for path_flag_col in ("echo_dcm_exists", "ecg_record_exists"):
        if path_flag_col in df.columns:
            summary[path_flag_col] = {
                str(k): int(v)
                for k, v in df[path_flag_col].value_counts(dropna=False).items()
            }

    main_labels = {}
    for candidate in MAIN_LABEL_CANDIDATES:
        if candidate in df.columns:
            series = pd.to_numeric(df[candidate], errors="coerce")
            main_labels[candidate] = {
                "missing_rate": float(series.isna().mean()),
                "positive_rate_nonnull": float(series.dropna().mean()) if series.dropna().shape[0] else None,
            }
    summary["main_label_candidates"] = main_labels
    return summary


def parse_timestamp(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, format="%Y/%m/%d %H:%M:%S", errors="coerce")


def build_multimodal_relationships(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    rel = df.copy()
    patient_col = get_patient_col(rel)
    pair_group_cols = [col for col in [patient_col, "echo_dcm_path", "ecg_record_path"] if col is not None]
    rel["pair_duplicate_count"] = (
        rel.groupby(pair_group_cols, dropna=False)[pair_group_cols[0]].transform("size")
    )
    rel["echo_pair_count"] = rel.groupby("echo_dcm_path", dropna=False)["echo_dcm_path"].transform("size")
    rel["ecg_pair_count"] = rel.groupby("ecg_record_path", dropna=False)["ecg_record_path"].transform("size")

    pairing_summary: dict[str, Any] = {
        "patient_id_column": patient_col,
        "subject_id_available": patient_col is not None,
        "time_fields_present": {
            "echo_time": "echo_time" in rel.columns,
            "ecg_time": "ecg_time" in rel.columns,
        },
    }

    if "echo_time" in rel.columns and "ecg_time" in rel.columns:
        rel["echo_time_parsed"] = parse_timestamp(rel["echo_time"])
        rel["ecg_time_parsed"] = parse_timestamp(rel["ecg_time"])
        rel["abs_time_diff_days_computed"] = (
            (rel["echo_time_parsed"] - rel["ecg_time_parsed"]).abs().dt.total_seconds() / 86400.0
        )
        valid = rel["abs_time_diff_days_computed"].dropna()
        if len(valid):
            p95 = float(valid.quantile(0.95))
            rel["time_gap_upper_tail_flag"] = (rel["abs_time_diff_days_computed"] > p95).astype(int)
            pairing_summary["time_diff_days"] = {
                "median": float(valid.median()),
                "p75": float(valid.quantile(0.75)),
                "p90": float(valid.quantile(0.90)),
                "p95": p95,
                "max": float(valid.max()),
            }
            pairing_summary["clinical_pairing_validation"] = (
                "Patient identity and ECG/Echo time difference can be audited from available fields, "
                "but encounter-level clinical pairing remains unverified."
            )
        else:
            pairing_summary["clinical_pairing_validation"] = (
                "Time fields exist but could not be parsed; only patient-level identity can be checked."
            )
    else:
        pairing_summary["clinical_pairing_validation"] = (
            "Current table lacks ECG/Echo time fields; only patient identity can be checked."
        )

    relationship_cols = [
        col
        for col in [
            patient_col,
            "echo_dcm_path",
            "ecg_record_path",
            "mapping_status",
            "pair_duplicate_count",
            "echo_pair_count",
            "ecg_pair_count",
            "echo_time",
            "ecg_time",
            "abs_time_diff_days_computed",
            "time_gap_upper_tail_flag",
        ]
        if col in rel.columns
    ]
    relationship_cols = [col for col in relationship_cols if col is not None]
    return rel[relationship_cols], pairing_summary


def choose_paths(paths: pd.Series, mode: str, sample_size: int, seed: int) -> list[str]:
    values = [str(x) for x in paths.dropna().unique().tolist()]
    if mode == "full" or len(values) <= sample_size:
        return sorted(values)
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(values), size=sample_size, replace=False)
    return sorted(values[idx] for idx in indices)


def audit_echo_files(
    df: pd.DataFrame,
    echo_root: Path,
    mode: str,
    sample_size: int,
    seed: int,
) -> pd.DataFrame:
    if "echo_dcm_path" not in df.columns:
        return pd.DataFrame()
    candidates = df["echo_dcm_path"]
    if "echo_dcm_exists" in df.columns:
        candidates = df.loc[pd.to_numeric(df["echo_dcm_exists"], errors="coerce") == 1, "echo_dcm_path"]
    stats = []
    for rel_path in choose_paths(candidates, mode=mode, sample_size=sample_size, seed=seed):
        row = inspect_dicom_file(echo_root / rel_path)
        row["relative_path"] = rel_path
        stats.append(row)
    return pd.DataFrame(stats)


def audit_ecg_files(
    df: pd.DataFrame,
    ecg_root: Path,
    mode: str,
    sample_size: int,
    seed: int,
) -> pd.DataFrame:
    if "ecg_record_path" not in df.columns:
        return pd.DataFrame()
    candidates = df["ecg_record_path"]
    if "ecg_record_exists" in df.columns:
        candidates = df.loc[pd.to_numeric(df["ecg_record_exists"], errors="coerce") == 1, "ecg_record_path"]
    stats = []
    for rel_path in choose_paths(candidates, mode=mode, sample_size=sample_size, seed=seed):
        row = inspect_wfdb_record(ecg_root / rel_path)
        row["relative_path"] = rel_path
        stats.append(row)
    return pd.DataFrame(stats)


def write_summary_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Data Audit 301",
        "",
        f"- mode: `{summary['file_audit_mode']}`",
        f"- sample_size: `{summary['sample_size']}`",
        "",
        "## CSV Summary",
    ]
    for name, table_summary in summary["tables"].items():
        lines.append(f"### {name}")
        for key, value in table_summary.items():
            lines.append(f"- {key}: `{value}`")
        lines.append("")
    lines.append("## Clinical Pairing")
    for key, value in summary["clinical_pairing"].items():
        lines.append(f"- {key}: `{value}`")
    lines.append("")
    lines.append("## Echo File Audit")
    for key, value in summary["echo_file_audit"].items():
        lines.append(f"- {key}: `{value}`")
    lines.append("")
    lines.append("## ECG File Audit")
    for key, value in summary["ecg_file_audit"].items():
        lines.append(f"- {key}: `{value}`")
    path.write_text("\n".join(lines), encoding="utf-8")


def filter_failures(df: pd.DataFrame, ok_col: str) -> pd.DataFrame:
    if df.empty or ok_col not in df.columns:
        return df.copy()
    return df[df[ok_col] != 1].copy()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    multimodal_df = pd.read_csv(args.multimodal_csv, low_memory=False)
    echo_df = pd.read_csv(args.echo_csv, low_memory=False)
    ecg_df = pd.read_csv(args.ecg_csv, low_memory=False)

    label_stats = pd.concat(
        [
            build_label_statistics(multimodal_df, "multimodal"),
            build_label_statistics(echo_df, "echo"),
            build_label_statistics(ecg_df, "ecg"),
        ],
        ignore_index=True,
    )
    label_stats.to_csv(output_dir / "label_statistics.csv", index=False)

    relationships_df, clinical_pairing = build_multimodal_relationships(multimodal_df)
    relationships_df.to_csv(output_dir / "multimodal_relationships.csv", index=False)

    echo_stats = audit_echo_files(
        echo_df,
        echo_root=Path(args.echo_root),
        mode=args.mode,
        sample_size=args.sample_size,
        seed=args.seed,
    )
    echo_stats.to_csv(output_dir / "echo_file_stats.csv", index=False)
    filter_failures(echo_stats, "pixel_array_ok").to_csv(output_dir / "echo_decode_failures.csv", index=False)

    ecg_stats = audit_ecg_files(
        ecg_df,
        ecg_root=Path(args.ecg_root),
        mode=args.mode,
        sample_size=args.sample_size,
        seed=args.seed,
    )
    ecg_stats.to_csv(output_dir / "ecg_file_stats.csv", index=False)
    filter_failures(ecg_stats, "rdrecord_ok").to_csv(output_dir / "ecg_read_failures.csv", index=False)

    summary = {
        "file_audit_mode": args.mode,
        "sample_size": args.sample_size,
        "tables": {
            "multimodal": summarize_table(multimodal_df, "multimodal", echo_col="echo_dcm_path", ecg_col="ecg_record_path"),
            "echo": summarize_table(echo_df, "echo", echo_col="echo_dcm_path"),
            "ecg": summarize_table(ecg_df, "ecg", ecg_col="ecg_record_path"),
        },
        "clinical_pairing": clinical_pairing,
        "echo_file_audit": {
            "files_checked": int(len(echo_stats)),
            "dcmread_ok": int(echo_stats["dcmread_ok"].sum()) if len(echo_stats) else 0,
            "pixel_array_ok": int(echo_stats["pixel_array_ok"].sum()) if len(echo_stats) else 0,
            "frame_count_distribution": (
                echo_stats["frame_count"].dropna().value_counts().head(10).to_dict() if "frame_count" in echo_stats else {}
            ),
            "resolution_distribution": (
                echo_stats.assign(res=lambda x: x["height"].astype("Int64").astype(str) + "x" + x["width"].astype("Int64").astype(str))["res"]
                .value_counts()
                .head(10)
                .to_dict()
                if len(echo_stats) and "height" in echo_stats and "width" in echo_stats
                else {}
            ),
            "photometric_distribution": (
                echo_stats["photometric_interpretation"].fillna("NA").value_counts().to_dict()
                if "photometric_interpretation" in echo_stats
                else {}
            ),
        },
        "ecg_file_audit": {
            "files_checked": int(len(ecg_stats)),
            "rdrecord_ok": int(ecg_stats["rdrecord_ok"].sum()) if len(ecg_stats) else 0,
            "fs_distribution": ecg_stats["fs"].dropna().value_counts().to_dict() if "fs" in ecg_stats else {},
            "n_sig_distribution": ecg_stats["n_sig"].dropna().value_counts().to_dict() if "n_sig" in ecg_stats else {},
            "sig_len_distribution": ecg_stats["sig_len"].dropna().value_counts().head(10).to_dict() if "sig_len" in ecg_stats else {},
        },
    }

    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_summary_markdown(output_dir / "summary.md", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
