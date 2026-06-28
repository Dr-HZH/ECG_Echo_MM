#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.build_hdf5_shards import STANDARD_12_LEADS, _to_echo_uint8
from utils.dicom_io import adapt_channel_count, read_dicom_pixels, resize_video_frames, sample_video_frames
from utils.hdf5_io import HDF5ShardReader, decode_wfdb_digital
from utils.wfdb_io import read_wfdb_record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare raw DICOM/WFDB reads against HDF5 shards.")
    parser.add_argument("--modality", choices=["echo", "ecg"], required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--echo_num_frames", type=int, default=32)
    parser.add_argument("--echo_image_size", type=int, default=224)
    parser.add_argument("--ecg_target_fs", type=float, default=500.0)
    parser.add_argument("--ecg_target_length", type=int, default=5000)
    parser.add_argument("--skip_raw", action="store_true")
    parser.add_argument("--raw_seconds_override", type=float, default=None)
    return parser.parse_args()


def read_raw_echo(path: Path, num_frames: int, image_size: int) -> np.ndarray:
    result = read_dicom_pixels(path)
    array = adapt_channel_count(result.array, target_channels=3)
    array = sample_video_frames(array, num_frames=num_frames, split="val")
    array = resize_video_frames(array, image_size=image_size)
    array = _to_echo_uint8(array)
    return np.ascontiguousarray(np.transpose(array, (1, 2, 3, 0)))


def read_raw_ecg(path: Path, target_fs: float, target_length: int) -> np.ndarray:
    result = read_wfdb_record(
        path,
        target_fs=target_fs,
        target_length=target_length,
        split="val",
        lead_order=STANDARD_12_LEADS,
        resample_if_needed=True,
    )
    return result.signal


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.manifest, low_memory=False)
    df = df[df["h5_path"].notna() & df["h5_key"].notna()].copy()
    if args.limit is not None:
        df = df.iloc[: args.limit].copy()
    path_col = "echo_dcm_path" if args.modality == "echo" else "ecg_record_path"
    data_root = Path(args.data_root)

    raw_arrays = []
    if args.skip_raw:
        if args.raw_seconds_override is None:
            raise ValueError("--skip_raw requires --raw_seconds_override.")
        raw_seconds = float(args.raw_seconds_override)
    else:
        raw_start = time.perf_counter()
        for _, row in df.iterrows():
            path = data_root / str(row[path_col])
            if args.modality == "echo":
                array = read_raw_echo(path, args.echo_num_frames, args.echo_image_size)
            else:
                array = read_raw_ecg(path, args.ecg_target_fs, args.ecg_target_length)
            raw_arrays.append(array)
        raw_seconds = time.perf_counter() - raw_start

    reader = HDF5ShardReader()
    h5_first = []
    h5_start = time.perf_counter()
    for _, row in df.iterrows():
        array, attrs = reader.read_with_attrs(row["h5_path"], row["h5_key"])
        if attrs.get("encoding") == "wfdb_digital":
            array = decode_wfdb_digital(array, attrs)
        h5_first.append(array)
    h5_first_seconds = time.perf_counter() - h5_start

    h5_start = time.perf_counter()
    for _, row in df.iterrows():
        array, attrs = reader.read_with_attrs(row["h5_path"], row["h5_key"])
        if attrs.get("encoding") == "wfdb_digital":
            decode_wfdb_digital(array, attrs)
    h5_warm_seconds = time.perf_counter() - h5_start
    reader.close()

    max_abs_errors = []
    mean_abs_errors = []
    nan_mask_mismatches = 0
    inf_mask_mismatches = 0
    for raw, h5_array in zip(raw_arrays, h5_first):
        raw_float = raw.astype(np.float32)
        h5_float = h5_array.astype(np.float32)
        nan_mask_mismatches += int(np.count_nonzero(np.isnan(raw_float) != np.isnan(h5_float)))
        inf_mask_mismatches += int(np.count_nonzero(np.isinf(raw_float) != np.isinf(h5_float)))
        finite = np.isfinite(raw_float) & np.isfinite(h5_float)
        if finite.any():
            delta = np.abs(raw_float[finite] - h5_float[finite])
            max_abs_errors.append(float(np.max(delta)))
            mean_abs_errors.append(float(np.mean(delta)))

    summary = {
        "modality": args.modality,
        "samples": len(df),
        "raw_seconds": raw_seconds,
        "h5_first_seconds": h5_first_seconds,
        "h5_warm_seconds": h5_warm_seconds,
        "raw_samples_per_second": len(df) / raw_seconds if raw_seconds else None,
        "h5_first_samples_per_second": len(df) / h5_first_seconds if h5_first_seconds else None,
        "h5_warm_samples_per_second": len(df) / h5_warm_seconds if h5_warm_seconds else None,
        "speedup_first": raw_seconds / h5_first_seconds if h5_first_seconds else None,
        "speedup_warm": raw_seconds / h5_warm_seconds if h5_warm_seconds else None,
        "max_abs_error": max(max_abs_errors, default=None),
        "mean_abs_error": float(np.mean(mean_abs_errors)) if mean_abs_errors else None,
        "nan_mask_mismatches": nan_mask_mismatches,
        "inf_mask_mismatches": inf_mask_mismatches,
        "raw_seconds_source": "override" if args.skip_raw else "measured_in_benchmark",
    }
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
