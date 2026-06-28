#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datasets.echo_dicom_dataset import EchoDicomDataset
from models.panecho_adapter import build_local_panecho_model


def parse_args():
    p = argparse.ArgumentParser(description="Zero-training PanEcho native-task evaluation.")
    p.add_argument("--manifest", default="manifests/301/echo_hfref_h5_manifest.csv")
    p.add_argument("--data_root", default="301/echo_only_dcm")
    p.add_argument("--input_format", choices=["dicom", "hdf5"], default="hdf5")
    p.add_argument("--checkpoint", default="checkpoints/pretrained/panecho.pt")
    p.add_argument("--repo", default="third_party/PanEcho")
    p.add_argument("--output_dir", default="outputs/panecho_zero_shot_769")
    p.add_argument("--device", default="cuda:4" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--limit", type=int)
    return p.parse_args()


def collate(batch):
    return {"echo": torch.stack([x["echo"] for x in batch]), "index": [x["row_index"] for x in batch]}


def bootstrap_auc(df, label, score, repetitions, seed=42):
    rng = np.random.default_rng(seed)
    patients = df.subject_id.drop_duplicates().to_numpy()
    values = []
    for _ in range(repetitions):
        sampled = rng.choice(patients, len(patients), replace=True)
        chunks = [df[df.subject_id == patient] for patient in sampled]
        boot = pd.concat(chunks, ignore_index=True)
        y = boot[label].to_numpy(dtype=int)
        if np.unique(y).size == 2:
            values.append(roc_auc_score(y, boot[score]))
    if not values:
        return {"lower": None, "upper": None, "valid": 0}
    return {"lower": float(np.percentile(values, 2.5)), "upper": float(np.percentile(values, 97.5)), "valid": len(values)}


def binary_metrics(df, label, score, bootstrap):
    valid = df[[label, score, "subject_id"]].dropna()
    y = valid[label].astype(int).to_numpy()
    if np.unique(y).size < 2:
        return {"n": len(valid), "positive": int(y.sum()), "auroc": None, "auprc": None,
                "patient_bootstrap_auroc_95ci": {"lower": None, "upper": None, "valid": 0}}
    return {
        "n": len(valid), "positive": int(y.sum()),
        "auroc": float(roc_auc_score(y, valid[score])),
        "auprc": float(average_precision_score(y, valid[score])),
        "patient_bootstrap_auroc_95ci": bootstrap_auc(valid, label, score, bootstrap),
    }


def main():
    args = parse_args()
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.manifest, low_memory=False)
    df = df[df.eligible_for_training == 1].reset_index(drop=True)
    if args.limit: df = df.head(args.limit).copy()
    dataset = EchoDicomDataset(
        args.manifest, dataframe=df, echo_path_col="echo_dcm_path", split="val",
        num_frames=16, image_size=224, target_channels=3, input_format=args.input_format,
        data_root=args.data_root, intensity_mode="uint8_scale",
        normalize_mean=[0.485, 0.456, 0.406], normalize_std=[0.229, 0.224, 0.225],
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                        shuffle=False, collate_fn=collate, pin_memory=True)
    device = torch.device(args.device)
    model = build_local_panecho_model(args.repo, args.checkpoint).to(device).eval()
    rows = []
    with torch.inference_mode():
        for batch in loader:
            pred = model(batch["echo"].to(device, non_blocking=True))
            rwma = pred["LVWallMotionAbnormalities"].flatten().cpu().numpy()
            ef = pred["EF"].flatten().cpu().numpy()
            lvf = pred["LVSystolicFunction"].cpu().numpy()
            for i, idx in enumerate(batch["index"]):
                rows.append((idx, rwma[i], ef[i], *lvf[i].tolist()))
    pred_df = pd.DataFrame(rows, columns=["row_index", "panecho_rwma_probability", "panecho_ef_prediction",
                                           "panecho_lvsf_0", "panecho_lvsf_1", "panecho_lvsf_2"])
    result = df.reset_index().rename(columns={"index": "row_index"}).merge(pred_df, on="row_index")
    result.to_csv(out / "predictions.csv", index=False)
    rwma = binary_metrics(result, "echo_rwma_feature_flag", "panecho_rwma_probability", args.bootstrap)
    proxy = binary_metrics(result, "echo_is_ischemic", "panecho_rwma_probability", args.bootstrap)
    ef_df = result[["ef", "panecho_ef_prediction"]].dropna().copy()
    ef_df["ef_percent"] = pd.to_numeric(ef_df.ef) * 100
    error = ef_df.panecho_ef_prediction - ef_df.ef_percent
    ef_metrics = {
        "n": len(ef_df), "mae": float(np.abs(error).mean()),
        "rmse": float(np.sqrt(np.mean(error ** 2))),
        "pearson": float(stats.pearsonr(ef_df.ef_percent, ef_df.panecho_ef_prediction).statistic),
        "spearman": float(stats.spearmanr(ef_df.ef_percent, ef_df.panecho_ef_prediction).statistic),
    }
    for name, payload in [("native_rwma_metrics.json", rwma), ("ischemia_proxy_metrics.json", proxy), ("ef_metrics.json", ef_metrics)]:
        (out / name).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (out / "report.md").write_text(
        "# PanEcho Zero-Training Evaluation\n\n"
        f"- Native RWMA AUROC: {rwma['auroc']}\n"
        f"- RWMA-based ischemia proxy AUROC: {proxy['auroc']}\n"
        f"- EF MAE: {ef_metrics['mae']:.4f}\n\n"
        "The ischemia analysis measures proxy discrimination, not direct ischemia classification.\n",
        encoding="utf-8",
    )
    print(json.dumps({"rwma": rwma, "ischemia_proxy": proxy, "ef": ef_metrics}, indent=2))


if __name__ == "__main__":
    main()
