#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from io import BytesIO
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.dicom_io import adapt_channel_count, read_dicom_pixels, resize_video_frames, sample_video_frames
from utils.wfdb_io import read_wfdb_record


STANDARD_12_LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build sharded HDF5 files from Echo DICOM or ECG WFDB records.")
    parser.add_argument("--modality", choices=["echo", "ecg"], required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--output_manifest", required=True)
    parser.add_argument("--summary_json", required=True)
    parser.add_argument("--failure_csv", required=True)
    parser.add_argument("--id_col", default=None)
    parser.add_argument("--path_col", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--shard_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--log_every", type=int, default=1)
    parser.add_argument("--compression", choices=["lzf", "gzip", "none"], default="lzf")
    parser.add_argument("--gzip_level", type=int, default=4)
    parser.add_argument("--echo_num_frames", type=int, default=32)
    parser.add_argument("--echo_image_size", type=int, default=224)
    parser.add_argument("--ecg_target_fs", type=float, default=500.0)
    parser.add_argument("--ecg_target_length", type=int, default=5000)
    parser.add_argument("--ecg_dtype", choices=["float32", "float16", "digital_int16"], default="float32")
    return parser.parse_args()


def _to_echo_uint8(array: np.ndarray) -> np.ndarray:
    if array.dtype == np.uint8:
        return np.ascontiguousarray(array)
    arr = array.astype(np.float32, copy=False)
    low = float(np.nanmin(arr))
    high = float(np.nanmax(arr))
    if high <= low:
        return np.zeros(arr.shape, dtype=np.uint8)
    arr = (arr - low) * (255.0 / (high - low))
    return np.clip(np.rint(arr), 0, 255).astype(np.uint8)


def _sample_indices(num_frames: int, target_frames: int) -> list[int]:
    if num_frames <= 0:
        return [0] * target_frames
    if num_frames == target_frames:
        return list(range(num_frames))
    if num_frames > target_frames:
        bins = np.linspace(0, num_frames, target_frames + 1)
        return [int((int(np.floor(left)) + max(int(np.ceil(right)) - 1, int(np.floor(left)))) // 2) for left, right in zip(bins[:-1], bins[1:])]
    return list(range(num_frames)) + [num_frames - 1] * (target_frames - num_frames)


def _read_jpeg_baseline_selected_frames(path: Path, num_frames: int, image_size: int) -> tuple[np.ndarray, dict[str, Any]]:
    import pydicom  # type: ignore
    from PIL import Image  # type: ignore
    from pydicom.encaps import generate_frames  # type: ignore

    dataset = pydicom.dcmread(str(path), force=True)
    transfer_syntax = str(getattr(getattr(dataset, "file_meta", None), "TransferSyntaxUID", ""))
    if transfer_syntax != "1.2.840.10008.1.2.4.50":
        raise ValueError(f"Unsupported fast-path transfer syntax: {transfer_syntax}")
    source_frames = int(getattr(dataset, "NumberOfFrames", 1) or 1)
    selected = _sample_indices(source_frames, num_frames)
    selected_set = set(selected)
    decoded: dict[int, np.ndarray] = {}
    for frame_idx, frame_bytes in enumerate(generate_frames(dataset.PixelData, number_of_frames=source_frames)):
        if frame_idx not in selected_set:
            continue
        image = Image.open(BytesIO(frame_bytes)).convert("RGB")
        if image.size != (image_size, image_size):
            image = image.resize((image_size, image_size), Image.Resampling.BILINEAR)
        decoded[frame_idx] = np.asarray(image, dtype=np.uint8)
        if len(decoded) == len(selected_set):
            break
    if not decoded:
        raise ValueError(f"No frames decoded from {path}")
    frames = [decoded.get(index, decoded[max(decoded.keys())]) for index in selected]
    array = np.ascontiguousarray(np.stack(frames, axis=0))
    attrs = {
        "source_path": str(path),
        "transfer_syntax_uid": transfer_syntax,
        "photometric_interpretation": str(getattr(dataset, "PhotometricInterpretation", "")),
        "source_frames": source_frames,
        "source_shape": f"{source_frames}x{int(getattr(dataset, 'Rows', 0) or 0)}x{int(getattr(dataset, 'Columns', 0) or 0)}x{int(getattr(dataset, 'SamplesPerPixel', 0) or 0)}",
        "source_height": int(getattr(dataset, "Rows", 0) or 0),
        "source_width": int(getattr(dataset, "Columns", 0) or 0),
        "source_channels": int(getattr(dataset, "SamplesPerPixel", 3) or 3),
        "finite_pixels": 1,
        "nonzero_fraction": float(np.count_nonzero(array) / array.size) if array.size else 0.0,
        "decode_fast_path": 1,
    }
    return array, attrs


def _process_echo(task: dict[str, Any]) -> dict[str, Any]:
    start = time.perf_counter()
    path = Path(task["abs_path"])
    try:
        try:
            array, attrs = _read_jpeg_baseline_selected_frames(
                path,
                num_frames=int(task["num_frames"]),
                image_size=int(task["image_size"]),
            )
        except Exception:
            result = read_dicom_pixels(path)
            array = adapt_channel_count(result.array, target_channels=3)
            source_shape = tuple(int(x) for x in array.shape)
            finite = bool(np.isfinite(array).all())
            nonzero_fraction = float(np.count_nonzero(array) / array.size) if array.size else 0.0
            array = sample_video_frames(array, num_frames=task["num_frames"], split="val")
            array = resize_video_frames(array, image_size=task["image_size"])
            array = _to_echo_uint8(array)
            array = np.ascontiguousarray(np.transpose(array, (1, 2, 3, 0)))
            attrs = {
                "source_path": str(path),
                "transfer_syntax_uid": result.transfer_syntax_uid or "",
                "photometric_interpretation": result.photometric_interpretation or "",
                "source_frames": result.frame_count,
                "source_shape": "x".join(str(x) for x in source_shape),
                "source_height": source_shape[-2],
                "source_width": source_shape[-1],
                "source_channels": source_shape[0],
                "finite_pixels": int(finite),
                "nonzero_fraction": nonzero_fraction,
                "decode_fast_path": 0,
            }
        return {
            "ok": True,
            "sample_id": task["sample_id"],
            "array": array,
            "source_bytes": path.stat().st_size,
            "seconds": time.perf_counter() - start,
            "attrs": attrs,
        }
    except Exception as exc:
        return {"ok": False, "sample_id": task["sample_id"], "error": str(exc), "seconds": time.perf_counter() - start}


def _process_ecg(task: dict[str, Any]) -> dict[str, Any]:
    start = time.perf_counter()
    path = Path(task["abs_path"])
    try:
        attrs: dict[str, Any]
        if task["dtype"] == "digital_int16":
            import wfdb  # type: ignore

            record = wfdb.rdrecord(str(path), physical=False)
            fs = float(record.fs)
            if fs != float(task["target_fs"]):
                raise ValueError(f"digital_int16 requires fs={task['target_fs']}, got {fs}")
            signal = np.asarray(record.d_signal).T
            names = [str(name) for name in record.sig_name]
            name_to_idx = {name: idx for idx, name in enumerate(names)}
            if any(lead not in name_to_idx for lead in STANDARD_12_LEADS):
                raise ValueError(f"Missing required leads: {sorted(set(STANDARD_12_LEADS) - set(names))}")
            indices = [name_to_idx[lead] for lead in STANDARD_12_LEADS]
            signal = signal[indices]
            gain = np.asarray(record.adc_gain, dtype=np.float32)[indices]
            baseline = np.asarray(record.baseline, dtype=np.int32)[indices]
            target_length = int(task["target_length"])
            if signal.shape[1] > target_length:
                start_idx = (signal.shape[1] - target_length) // 2
                signal = signal[:, start_idx : start_idx + target_length]
            elif signal.shape[1] < target_length:
                pad = np.repeat(baseline[:, None], target_length - signal.shape[1], axis=1)
                signal = np.concatenate([signal, pad], axis=1)
            if signal.min() < np.iinfo(np.int16).min or signal.max() > np.iinfo(np.int16).max:
                raise ValueError("WFDB digital values exceed int16 range.")
            array = np.ascontiguousarray(signal.astype(np.int16))
            attrs = {
                "source_path": str(path),
                "fs": fs,
                "lead_order": "|".join(STANDARD_12_LEADS),
                "missing_leads": "",
                "encoding": "wfdb_digital",
                "adc_gain": gain,
                "baseline": baseline,
                "digital_nan": -32768,
            }
        else:
            result = read_wfdb_record(
                path,
                target_fs=task["target_fs"],
                target_length=task["target_length"],
                split="val",
                lead_order=STANDARD_12_LEADS,
                resample_if_needed=True,
            )
            dtype = np.float16 if task["dtype"] == "float16" else np.float32
            array = np.ascontiguousarray(result.signal.astype(dtype))
            attrs = {
                "source_path": str(path),
                "fs": result.fs,
                "lead_order": "|".join(STANDARD_12_LEADS),
                "missing_leads": "|".join(result.missing_leads),
                "encoding": task["dtype"],
            }
        source_bytes = 0
        for suffix in (".dat", ".hea"):
            source_file = path.with_suffix(suffix)
            if source_file.exists():
                source_bytes += source_file.stat().st_size
        return {
            "ok": True,
            "sample_id": task["sample_id"],
            "array": array,
            "source_bytes": source_bytes,
            "seconds": time.perf_counter() - start,
            "attrs": attrs,
        }
    except Exception as exc:
        return {"ok": False, "sample_id": task["sample_id"], "error": str(exc), "seconds": time.perf_counter() - start}


def _iter_results(tasks: list[dict[str, Any]], modality: str, num_workers: int) -> Iterable[dict[str, Any]]:
    worker = _process_echo if modality == "echo" else _process_ecg
    if num_workers <= 1:
        yield from map(worker, tasks)
        return
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        yield from executor.map(worker, tasks, chunksize=1)


def _dataset_options(args: argparse.Namespace, array: np.ndarray) -> dict[str, Any]:
    if args.compression == "none":
        return {}
    options: dict[str, Any] = {"compression": args.compression, "shuffle": True}
    if args.compression == "gzip":
        options["compression_opts"] = args.gzip_level
    if args.modality == "echo":
        options["chunks"] = (1, array.shape[1], array.shape[2], array.shape[3])
    else:
        options["chunks"] = (array.shape[0], min(1000, array.shape[1]))
    return options


def main() -> None:
    args = parse_args()
    try:
        import h5py  # type: ignore
    except Exception as exc:
        raise RuntimeError("h5py is required to build HDF5 shards.") from exc

    manifest = pd.read_csv(args.manifest, low_memory=False)
    if "eligible_for_training" in manifest.columns:
        manifest = manifest[manifest["eligible_for_training"] == 1].copy()
    if args.offset:
        manifest = manifest.iloc[args.offset :].copy()
    if args.limit is not None:
        manifest = manifest.iloc[: args.limit].copy()
    manifest = manifest.reset_index(drop=True)

    id_col = args.id_col or ("echo_id" if args.modality == "echo" else "ecg_id")
    path_col = args.path_col or ("echo_dcm_path" if args.modality == "echo" else "ecg_record_path")
    data_root = Path(args.data_root)
    tasks = []
    for _, row in manifest.iterrows():
        base = {
            "sample_id": str(row[id_col]),
            "abs_path": str(data_root / str(row[path_col])),
        }
        if args.modality == "echo":
            base.update(num_frames=args.echo_num_frames, image_size=args.echo_image_size)
        else:
            base.update(target_fs=args.ecg_target_fs, target_length=args.ecg_target_length, dtype=args.ecg_dtype)
        tasks.append(base)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    Path(args.output_manifest).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.failure_csv).parent.mkdir(parents=True, exist_ok=True)
    group_name = "videos" if args.modality == "echo" else "signals"
    compression_name = args.compression
    output_rows = []
    failures = []
    source_bytes = 0
    processing_seconds = 0.0
    build_start = time.perf_counter()
    current_h5 = None
    current_shard = -1

    try:
        for result in _iter_results(tasks, args.modality, args.num_workers):
            processing_seconds += float(result["seconds"])
            if not result["ok"]:
                failures.append({"sample_id": result["sample_id"], "error": result["error"]})
                print(json.dumps({"event": "h5_error", **failures[-1]}), flush=True)
                continue

            success_index = len(output_rows)
            shard_index = success_index // args.shard_size
            if shard_index != current_shard:
                if current_h5 is not None:
                    current_h5.close()
                shard_path = output_dir / f"shard_{shard_index:05d}.h5"
                current_h5 = h5py.File(shard_path, "w", libver="latest")
                current_h5.attrs["schema_version"] = "1"
                current_h5.attrs["modality"] = args.modality
                current_h5.attrs["compression"] = compression_name
                current_h5.require_group(group_name)
                current_shard = shard_index

            array = result["array"]
            key = f"/{group_name}/{result['sample_id']}"
            dataset = current_h5.create_dataset(key, data=array, **_dataset_options(args, array))
            for attr_key, attr_value in result["attrs"].items():
                dataset.attrs[attr_key] = attr_value
            source_bytes += int(result["source_bytes"])
            output_rows.append(
                {
                    id_col: result["sample_id"],
                    "h5_path": str(output_dir / f"shard_{shard_index:05d}.h5"),
                    "h5_key": key,
                    "h5_shape": "x".join(str(x) for x in array.shape),
                    "h5_dtype": str(array.dtype),
                    "readable": 1,
                    "num_frames": int(result["attrs"].get("source_frames", 0)),
                    "height": int(result["attrs"].get("source_height", 0)),
                    "width": int(result["attrs"].get("source_width", 0)),
                    "channels": int(result["attrs"].get("source_channels", 0)),
                    "photometric_interpretation": result["attrs"].get("photometric_interpretation", ""),
                    "finite_pixels": int(result["attrs"].get("finite_pixels", 0)),
                    "nonzero_fraction": float(result["attrs"].get("nonzero_fraction", 0.0)),
                    "decode_seconds": float(result["seconds"]),
                }
            )
            if len(output_rows) % args.log_every == 0 or len(output_rows) == len(tasks):
                print(
                    json.dumps({"event": "h5_item", "count": len(output_rows), "sample_id": result["sample_id"]}),
                    flush=True,
                )
    finally:
        if current_h5 is not None:
            current_h5.close()

    mapping = pd.DataFrame(output_rows)
    output_manifest = manifest.merge(mapping, on=id_col, how="left")
    output_manifest.to_csv(args.output_manifest, index=False)
    pd.DataFrame(failures, columns=["sample_id", "error"]).to_csv(args.failure_csv, index=False)
    h5_bytes = sum(path.stat().st_size for path in output_dir.glob("shard_*.h5"))
    summary = {
        "modality": args.modality,
        "requested": len(tasks),
        "written": len(output_rows),
        "failed": len(failures),
        "source_bytes": source_bytes,
        "h5_bytes": h5_bytes,
        "h5_to_source_ratio": h5_bytes / source_bytes if source_bytes else None,
        "wall_seconds": time.perf_counter() - build_start,
        "sum_worker_seconds": processing_seconds,
        "compression": compression_name,
        "shard_size": args.shard_size,
        "num_workers": args.num_workers,
    }
    Path(args.summary_json).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"event": "h5_complete", **summary}, indent=2), flush=True)
    if failures:
        raise RuntimeError(f"HDF5 build completed with {len(failures)} failures; see {args.failure_csv}")


if __name__ == "__main__":
    main()
