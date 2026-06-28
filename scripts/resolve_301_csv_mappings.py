#!/usr/bin/env python3

import argparse
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path


def set_csv_field_limit() -> None:
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


def strip_or_none(value: str | None) -> str | None:
    if value is None:
        return None
    value = value.strip()
    return value or None


def echo_dcm_from_npy(npy_rel: str | None) -> str | None:
    npy_rel = strip_or_none(npy_rel)
    if not npy_rel:
        return None
    patterns = (
        r"_\d+_a4c\.npy$",
        r"_a4c_\d+\.npy$",
    )
    for pattern in patterns:
        if re.search(pattern, npy_rel):
            return re.sub(pattern, ".dcm", npy_rel)
    if npy_rel.endswith(".npy"):
        return npy_rel[:-4] + ".dcm"
    return None


def ecg_record_from_npy(npy_rel: str | None) -> str | None:
    npy_rel = strip_or_none(npy_rel)
    if not npy_rel or not npy_rel.endswith(".npy"):
        return None
    return npy_rel[:-4]


def path_exists(root: Path, rel_path: str | None) -> bool:
    if not rel_path:
        return False
    return (root / rel_path).exists()


def resolve_multimodal_row(row: dict[str, str], base_dir: Path, stats: Counter) -> dict[str, str]:
    echo_root = base_dir / "echo_only_dcm"
    ecg_root = base_dir / "ecg-diagnostic-electrocardiogram-matched-subset"

    echo_npy_path = strip_or_none(row.get("echo_npy_path"))
    ecg_npy_path = strip_or_none(row.get("ecg_npy_path"))

    echo_dcm_path = echo_dcm_from_npy(echo_npy_path)
    ecg_record_path = ecg_record_from_npy(ecg_npy_path)
    ecg_dat_path = f"{ecg_record_path}.dat" if ecg_record_path else None
    ecg_hea_path = f"{ecg_record_path}.hea" if ecg_record_path else None

    echo_exists = path_exists(echo_root, echo_dcm_path)
    ecg_dat_exists = path_exists(ecg_root, ecg_dat_path)
    ecg_hea_exists = path_exists(ecg_root, ecg_hea_path)
    ecg_record_exists = ecg_dat_exists and ecg_hea_exists

    original_ecg_file_path = strip_or_none(row.get("ecg_file_path"))
    ecg_file_path_filled = False
    if not original_ecg_file_path and ecg_record_path:
        row["ecg_file_path"] = ecg_record_path
        ecg_file_path_filled = True
        stats["filled_ecg_file_path"] += 1

    if not strip_or_none(row.get("echo_file_paths")):
        stats["blank_echo_file_paths"] += 1

    row["echo_dcm_path"] = echo_dcm_path or ""
    row["echo_dcm_exists"] = str(int(echo_exists))
    row["ecg_record_path"] = ecg_record_path or ""
    row["ecg_dat_path"] = ecg_dat_path or ""
    row["ecg_hea_path"] = ecg_hea_path or ""
    row["ecg_record_exists"] = str(int(ecg_record_exists))
    row["ecg_file_path_was_filled"] = str(int(ecg_file_path_filled))

    if echo_exists and ecg_record_exists:
        row["mapping_status"] = "ok"
        stats["ok"] += 1
    else:
        missing = []
        if not echo_exists:
            missing.append("echo_dcm")
        if not ecg_dat_exists:
            missing.append("ecg_dat")
        if not ecg_hea_exists:
            missing.append("ecg_hea")
        row["mapping_status"] = "missing:" + ",".join(missing)
        stats["missing"] += 1

    stats["rows"] += 1
    return row


def resolve_echo_row(row: dict[str, str], base_dir: Path, stats: Counter) -> dict[str, str]:
    echo_root = base_dir / "echo_only_dcm"
    npy_path = strip_or_none(row.get("npy_path"))
    echo_dcm_path = echo_dcm_from_npy(npy_path)
    echo_exists = path_exists(echo_root, echo_dcm_path)

    row["echo_dcm_path"] = echo_dcm_path or ""
    row["echo_dcm_exists"] = str(int(echo_exists))
    row["mapping_status"] = "ok" if echo_exists else "missing:echo_dcm"

    stats["rows"] += 1
    stats["ok" if echo_exists else "missing"] += 1
    return row


def resolve_ecg_row(row: dict[str, str], base_dir: Path, stats: Counter) -> dict[str, str]:
    ecg_root = base_dir / "ecg-diagnostic-electrocardiogram-matched-subset"
    npy_path = strip_or_none(row.get("npy_path"))
    ecg_record_path = ecg_record_from_npy(npy_path)
    ecg_dat_path = f"{ecg_record_path}.dat" if ecg_record_path else None
    ecg_hea_path = f"{ecg_record_path}.hea" if ecg_record_path else None

    dat_exists = path_exists(ecg_root, ecg_dat_path)
    hea_exists = path_exists(ecg_root, ecg_hea_path)
    record_exists = dat_exists and hea_exists

    original_ecg_file_path = strip_or_none(row.get("ecg_file_path"))
    ecg_file_path_filled = False
    if not original_ecg_file_path and ecg_record_path:
        row["ecg_file_path"] = ecg_record_path
        ecg_file_path_filled = True
        stats["filled_ecg_file_path"] += 1

    row["ecg_record_path"] = ecg_record_path or ""
    row["ecg_dat_path"] = ecg_dat_path or ""
    row["ecg_hea_path"] = ecg_hea_path or ""
    row["ecg_record_exists"] = str(int(record_exists))
    row["ecg_file_path_was_filled"] = str(int(ecg_file_path_filled))
    row["mapping_status"] = "ok" if record_exists else "missing:ecg_dat_or_hea"

    stats["rows"] += 1
    stats["ok" if record_exists else "missing"] += 1
    return row


def resolve_csv(path: Path, base_dir: Path, output_dir: Path) -> dict[str, object]:
    stats: Counter = Counter()
    if path.name == "multimodal_pairs_relative.csv":
        resolver = resolve_multimodal_row
        extras = [
            "echo_dcm_path",
            "echo_dcm_exists",
            "ecg_record_path",
            "ecg_dat_path",
            "ecg_hea_path",
            "ecg_record_exists",
            "ecg_file_path_was_filled",
            "mapping_status",
        ]
    elif path.name == "echo_allef_relative.csv":
        resolver = resolve_echo_row
        extras = [
            "echo_dcm_path",
            "echo_dcm_exists",
            "mapping_status",
        ]
    elif path.name == "ecg_hfref_relative.csv":
        resolver = resolve_ecg_row
        extras = [
            "ecg_record_path",
            "ecg_dat_path",
            "ecg_hea_path",
            "ecg_record_exists",
            "ecg_file_path_was_filled",
            "mapping_status",
        ]
    else:
        raise ValueError(f"Unsupported CSV: {path.name}")

    output_path = output_dir / f"{path.stem}_resolved.csv"
    with path.open("r", newline="") as src:
        reader = csv.DictReader(src)
        fieldnames = list(reader.fieldnames or [])
        for extra in extras:
            if extra not in fieldnames:
                fieldnames.append(extra)

        with output_path.open("w", newline="") as dst:
            writer = csv.DictWriter(dst, fieldnames=fieldnames)
            writer.writeheader()
            for row in reader:
                writer.writerow(resolver(row, base_dir, stats))

    summary = {
        "input_csv": str(path),
        "output_csv": str(output_path),
        "rows": stats["rows"],
        "ok": stats["ok"],
        "missing": stats["missing"],
    }
    if "filled_ecg_file_path" in stats:
        summary["filled_ecg_file_path"] = stats["filled_ecg_file_path"]
    if "blank_echo_file_paths" in stats:
        summary["blank_echo_file_paths"] = stats["blank_echo_file_paths"]
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Resolve CSV path mappings for dataset 301.")
    parser.add_argument(
        "--base-dir",
        default="/media/data1/huangzihao/ECG_Echo_MM/301",
        help="Dataset base directory that contains the three CSV files.",
    )
    parser.add_argument(
        "--output-dir",
        default="/media/data1/huangzihao/ECG_Echo_MM/resolved_csvs/301",
        help="Writable directory for resolved CSVs and summary JSON.",
    )
    parser.add_argument(
        "--csvs",
        nargs="*",
        default=[
            "multimodal_pairs_relative.csv",
            "echo_allef_relative.csv",
            "ecg_hfref_relative.csv",
        ],
        help="CSV file names relative to --base-dir.",
    )
    args = parser.parse_args()

    set_csv_field_limit()
    base_dir = Path(args.base_dir).resolve()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    for csv_name in args.csvs:
        csv_path = base_dir / csv_name
        if not csv_path.exists():
            raise FileNotFoundError(csv_path)
        summaries.append(resolve_csv(csv_path, base_dir, output_dir))

    summary_path = output_dir / "csv_resolution_summary.json"
    summary_path.write_text(json.dumps(summaries, indent=2), encoding="utf-8")

    print(json.dumps(summaries, indent=2))
    print(f"summary_json={summary_path}")


if __name__ == "__main__":
    main()
