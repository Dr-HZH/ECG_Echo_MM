#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Resumable full Echo HDF5 conversion in independent parts.")
    parser.add_argument("--manifest", default="manifests/301/echo_hfref_manifest.csv")
    parser.add_argument("--data_root", default="301/echo_only_dcm")
    parser.add_argument("--output_root", default="data_h5/echo_hfref_t32_224_lzf")
    parser.add_argument("--output_manifest", default="manifests/301/echo_hfref_h5_manifest.csv")
    parser.add_argument("--summary_json", default="reports/hdf5_echo_full_summary.json")
    parser.add_argument("--failure_csv", default="reports/hdf5_echo_full_failures.csv")
    parser.add_argument("--part_size", type=int, default=64)
    parser.add_argument("--id_col", default="echo_id")
    parser.add_argument("--num_workers", type=int, default=1)
    return parser.parse_args()


def part_is_complete(summary_path: Path, requested: int) -> bool:
    if not summary_path.exists():
        return False
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    return summary.get("written") == requested and summary.get("failed") == 0


def main() -> None:
    args = parse_args()
    source = pd.read_csv(args.manifest, low_memory=False)
    if "eligible_for_training" in source.columns:
        source = source[source["eligible_for_training"] == 1].copy()
    source = source.reset_index(drop=True)

    output_root = Path(args.output_root)
    parts_root = output_root / "parts"
    parts_root.mkdir(parents=True, exist_ok=True)
    summaries = []
    part_manifests = []
    part_failures = []

    for part_index, offset in enumerate(range(0, len(source), args.part_size)):
        requested = min(args.part_size, len(source) - offset)
        part_dir = output_root / f"part_{part_index:03d}"
        part_manifest = parts_root / f"part_{part_index:03d}.csv"
        part_summary = parts_root / f"part_{part_index:03d}_summary.json"
        part_failure = parts_root / f"part_{part_index:03d}_failures.csv"
        if not part_is_complete(part_summary, requested):
            command = [
                sys.executable,
                "scripts/build_hdf5_shards.py",
                "--modality", "echo",
                "--manifest", args.manifest,
                "--data_root", args.data_root,
                "--output_dir", str(part_dir),
                "--output_manifest", str(part_manifest),
                "--summary_json", str(part_summary),
                "--failure_csv", str(part_failure),
                "--id_col", args.id_col,
                "--offset", str(offset),
                "--limit", str(requested),
                "--shard_size", str(args.part_size),
                "--num_workers", str(args.num_workers),
                "--log_every", "16",
                "--compression", "lzf",
                "--echo_num_frames", "32",
                "--echo_image_size", "224",
            ]
            print(json.dumps({"event": "part_start", "part": part_index, "offset": offset, "count": requested}), flush=True)
            subprocess.run(command, check=True)
        else:
            print(json.dumps({"event": "part_skip", "part": part_index, "reason": "already_complete"}), flush=True)

        summaries.append(json.loads(part_summary.read_text(encoding="utf-8")))
        part_manifests.append(pd.read_csv(part_manifest, low_memory=False))
        if part_failure.exists():
            failures = pd.read_csv(part_failure, low_memory=False)
            if not failures.empty:
                failures["part"] = part_index
                part_failures.append(failures)

    combined_parts = pd.concat(part_manifests, ignore_index=True)
    audit_cols = ["h5_path", "h5_key", "h5_shape", "h5_dtype", "readable", "num_frames",
                  "height", "width", "channels", "photometric_interpretation", "finite_pixels",
                  "nonzero_fraction", "decode_seconds"]
    mapping_cols = [args.id_col, *audit_cols]
    mapping = combined_parts[mapping_cols].dropna(subset=["h5_key"]).drop_duplicates(args.id_col)
    final_manifest = source.drop(columns=[col for col in mapping_cols[1:] if col in source.columns]).merge(
        mapping, on=args.id_col, how="left", validate="one_to_one"
    )
    Path(args.output_manifest).parent.mkdir(parents=True, exist_ok=True)
    final_manifest.to_csv(args.output_manifest, index=False)

    all_failures = pd.concat(part_failures, ignore_index=True) if part_failures else pd.DataFrame(columns=["sample_id", "error", "part"])
    Path(args.failure_csv).parent.mkdir(parents=True, exist_ok=True)
    all_failures.to_csv(args.failure_csv, index=False)
    aggregate = {
        "modality": "echo",
        "requested": int(len(source)),
        "written": int(final_manifest["h5_key"].notna().sum()),
        "failed": int(len(all_failures)),
        "source_bytes": int(sum(item["source_bytes"] for item in summaries)),
        "h5_bytes": int(sum(item["h5_bytes"] for item in summaries)),
        "wall_seconds_parts": float(sum(item["wall_seconds"] for item in summaries)),
        "parts": len(summaries),
        "part_size": args.part_size,
    }
    aggregate["h5_to_source_ratio"] = aggregate["h5_bytes"] / aggregate["source_bytes"]
    Path(args.summary_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary_json).write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    print(json.dumps({"event": "full_echo_h5_complete", **aggregate}, indent=2), flush=True)
    if aggregate["written"] != aggregate["requested"] or aggregate["failed"]:
        raise RuntimeError("Full Echo HDF5 conversion is incomplete.")


if __name__ == "__main__":
    main()
