#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets.racafuse_dataset import RACaFuseDataset
from losses.ranking_calibration_losses import RACaFuseLoss
from models.racafuse import RACaFuse
from utils.metrics import compute_binary_metrics, find_best_threshold


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train RA-CaFuse with patient-level 5-fold CV.")
    parser.add_argument("--config", default="configs/racafuse_from_scratch.yaml")
    parser.add_argument("--folds", default=None, help="Comma-separated folds to run, e.g. 1,2,3.")
    parser.add_argument("--rebuild_manifest", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def maybe_build_manifest(config: dict[str, Any], force: bool = False) -> None:
    data = config["data"]
    manifest = Path(data["manifest_csv"])
    split = Path(data["split_csv"])
    if manifest.exists() and split.exists() and not force:
        return
    cmd = [
        sys.executable,
        "scripts/build_racafuse_manifest.py",
        "--pairs_csv",
        data["pairs_csv"],
        "--echo_manifest",
        data["echo_manifest"],
        "--ecg_h5_manifest",
        data["ecg_h5_manifest"],
        "--echo_h5_dir",
        data["echo_h5_dir"],
        "--output_manifest",
        data["manifest_csv"],
        "--output_split",
        data["split_csv"],
        "--summary_json",
        data["manifest_summary_json"],
        "--max_days",
        str(data.get("max_days", 30.0)),
        "--label_col",
        data.get("source_label_col", "echo_strict_is_ischemic"),
        "--n_splits",
        str(config["cv"].get("n_splits", 5)),
        "--inner_val_ratio",
        str(config["cv"].get("inner_val_ratio", 0.1)),
        "--seed",
        str(config["seed"]),
    ]
    subprocess.run(cmd, check=True)


def make_model(config: dict[str, Any]) -> nn.Module:
    model_cfg = config["model"]
    return RACaFuse(
        ecg_in_channels=model_cfg.get("ecg_in_channels", 12),
        echo_in_channels=model_cfg.get("echo_in_channels", 3),
        embedding_dim=model_cfg.get("embedding_dim", 256),
        ecg_base_channels=model_cfg.get("ecg_base_channels", 48),
        echo_base_channels=model_cfg.get("echo_base_channels", 24),
        dropout=model_cfg.get("dropout", 0.1),
        meta_dim=model_cfg.get("meta_dim", 3),
    )


def make_dataset(config: dict[str, Any], dataframe: pd.DataFrame, split: str) -> RACaFuseDataset:
    ds_cfg = config["dataset"]
    return RACaFuseDataset(
        csv_path=config["data"]["manifest_csv"],
        dataframe=dataframe,
        split=split,
        data_root=config["data"].get("data_root", "."),
        echo_num_frames=ds_cfg.get("echo_num_frames", 16),
        echo_image_size=ds_cfg.get("echo_image_size", 112),
        echo_channels=ds_cfg.get("echo_channels", 3),
        echo_intensity_mode=ds_cfg.get("echo_intensity_mode", "uint8_scale"),
        echo_crop_scale_range=tuple(ds_cfg["echo_crop_scale_range"]) if ds_cfg.get("echo_crop_scale_range") else None,
        echo_brightness_jitter=ds_cfg.get("echo_brightness_jitter", 0.0),
        echo_contrast_jitter=ds_cfg.get("echo_contrast_jitter", 0.0),
        echo_translate_fraction=ds_cfg.get("echo_translate_fraction", 0.0),
        echo_normalize_mean=ds_cfg.get("echo_normalize_mean"),
        echo_normalize_std=ds_cfg.get("echo_normalize_std"),
        ecg_target_length=ds_cfg.get("ecg_target_length", 5000),
        label_col="label",
    )


def make_loader(config: dict[str, Any], dataframe: pd.DataFrame, split: str, shuffle: bool) -> DataLoader:
    loader_cfg = config["loader"]
    return DataLoader(
        make_dataset(config, dataframe, split=split),
        batch_size=loader_cfg.get("batch_size", 8),
        shuffle=shuffle,
        num_workers=loader_cfg.get("num_workers", 0),
        pin_memory=loader_cfg.get("pin_memory", True),
        persistent_workers=loader_cfg.get("num_workers", 0) > 0,
        drop_last=shuffle,
    )


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    out = dict(batch)
    for key in ("ecg", "echo", "meta", "label"):
        out[key] = batch[key].to(device, non_blocking=True)
    return out


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: torch.device, use_calibrated: bool) -> pd.DataFrame:
    model.eval()
    rows: list[dict[str, Any]] = []
    for batch in loader:
        moved = move_batch(batch, device)
        output = model(moved["ecg"], moved["echo"], moved["meta"])
        logit_key = "calibrated_logit" if use_calibrated else "raw_logit"
        probs = torch.sigmoid(output[logit_key]).detach().cpu().numpy()
        raw_probs = torch.sigmoid(output["raw_logit"]).detach().cpu().numpy()
        labels = moved["label"].detach().cpu().numpy()
        alpha_ecg = output["alpha_ecg"].detach().cpu().numpy()
        alpha_echo = output["alpha_echo"].detach().cpu().numpy()
        temps = output["temperature"].detach().cpu().numpy()
        for idx, sample_id in enumerate(batch["sample_id"]):
            rows.append(
                {
                    "sample_id": sample_id,
                    "patient_id": batch["patient_id"][idx],
                    "label": float(labels[idx]),
                    "probability": float(probs[idx]),
                    "raw_probability": float(raw_probs[idx]),
                    "alpha_ecg": float(alpha_ecg[idx]),
                    "alpha_echo": float(alpha_echo[idx]),
                    "temperature": float(temps[idx]),
                    "time_delta_days": float(batch["time_delta_days"][idx]),
                }
            )
    return pd.DataFrame(rows)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: RACaFuseLoss,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    use_calibrated_loss: bool,
    amp: bool,
) -> dict[str, float]:
    model.train()
    totals = {"loss": 0.0, "focal": 0.0, "auc": 0.0, "brier": 0.0}
    n = 0
    for batch in loader:
        moved = move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=amp):
            output = model(moved["ecg"], moved["echo"], moved["meta"])
            logits = output["calibrated_logit"] if use_calibrated_loss else output["raw_logit"]
            losses = criterion(logits, moved["label"])
        scaler.scale(losses["loss"]).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        scaler.step(optimizer)
        scaler.update()
        batch_n = moved["label"].numel()
        n += batch_n
        for key in totals:
            totals[key] += float(losses[key].detach().cpu()) * batch_n
    return {key: value / max(n, 1) for key, value in totals.items()}


def evaluate_split(preds: pd.DataFrame, threshold: float | None = None) -> tuple[dict[str, float], float]:
    y_true = preds["label"].to_numpy(dtype=int)
    y_prob = preds["probability"].to_numpy(dtype=float)
    if threshold is None:
        threshold = find_best_threshold(y_true, y_prob, strategy="f1")
    metrics = compute_binary_metrics(y_true, y_prob, threshold=threshold)
    balanced_accuracy = (metrics.sensitivity + metrics.specificity) / 2.0
    return {
        "roc_auc": metrics.auroc,
        "pr_auc": metrics.auprc,
        "f1": metrics.f1,
        "accuracy": metrics.accuracy,
        "balanced_accuracy": balanced_accuracy,
        "sensitivity": metrics.sensitivity,
        "specificity": metrics.specificity,
        "threshold": threshold,
    }, threshold


def run_fold(config: dict[str, Any], fold: int, manifest: pd.DataFrame, split_df: pd.DataFrame, device: torch.device) -> dict[str, Any]:
    fold_dir = Path(config["output_dir"]) / f"fold_{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    train_ids = split_df[(split_df["fold"] == fold) & (split_df["split"] == "train")]["sample_id"]
    val_ids = split_df[(split_df["fold"] == fold) & (split_df["split"] == "inner_val")]["sample_id"]
    test_ids = split_df[(split_df["fold"] == fold) & (split_df["split"] == "outer_test")]["sample_id"]
    train_df = manifest[manifest["sample_id"].isin(train_ids)].reset_index(drop=True)
    val_df = manifest[manifest["sample_id"].isin(val_ids)].reset_index(drop=True)
    test_df = manifest[manifest["sample_id"].isin(test_ids)].reset_index(drop=True)

    model = make_model(config).to(device)
    train_pos = float((train_df["label"] == 1).sum())
    train_neg = float((train_df["label"] == 0).sum())
    pos_weight = train_neg / max(train_pos, 1.0)
    loss_cfg = config["loss"]
    criterion = RACaFuseLoss(
        focal_alpha=loss_cfg.get("focal_alpha", 0.5),
        focal_gamma=loss_cfg.get("focal_gamma", 2.0),
        pos_weight=pos_weight if loss_cfg.get("use_pos_weight", True) else None,
        lambda_auc=loss_cfg.get("lambda_auc", 0.1),
        lambda_brier=loss_cfg.get("lambda_brier", 0.05),
        auc_margin=loss_cfg.get("auc_margin", 0.2),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["train"].get("lr", 1e-3),
        weight_decay=config["train"].get("weight_decay", 1e-4),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["train"].get("epochs", 10))
    amp = bool(config["train"].get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler(device=device.type, enabled=amp)
    train_loader = make_loader(config, train_df, split="train", shuffle=True)
    val_loader = make_loader(config, val_df, split="val", shuffle=False)
    test_loader = make_loader(config, test_df, split="val", shuffle=False)

    best = {"epoch": -1, "val_roc_auc": -math.inf, "val_threshold": 0.5}
    patience = int(config["train"].get("patience", 4))
    stale = 0
    history = []
    for epoch in range(1, int(config["train"].get("epochs", 10)) + 1):
        start = time.time()
        train_losses = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scaler,
            device,
            use_calibrated_loss=bool(config["train"].get("use_calibrated_loss", True)),
            amp=amp,
        )
        scheduler.step()
        val_preds = predict(model, val_loader, device, use_calibrated=bool(config["eval"].get("use_calibrated", True)))
        val_metrics, val_threshold = evaluate_split(val_preds, threshold=None)
        row = {"epoch": epoch, "seconds": time.time() - start, **train_losses, **{f"val_{k}": v for k, v in val_metrics.items()}}
        history.append(row)
        print(json.dumps({"fold": fold, **row}, ensure_ascii=False), flush=True)
        if val_metrics["roc_auc"] > best["val_roc_auc"]:
            best = {"epoch": epoch, "val_roc_auc": val_metrics["roc_auc"], "val_threshold": val_threshold}
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": config,
                    "fold": fold,
                    "epoch": epoch,
                    "val_metrics": val_metrics,
                    "pos_weight": pos_weight,
                },
                fold_dir / "best.ckpt",
            )
            val_preds.to_csv(fold_dir / "inner_val_predictions.csv", index=False)
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break

    checkpoint = torch.load(fold_dir / "best.ckpt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_preds = predict(model, test_loader, device, use_calibrated=bool(config["eval"].get("use_calibrated", True)))
    test_metrics, _ = evaluate_split(test_preds, threshold=float(best["val_threshold"]))
    test_preds["fold"] = fold
    test_preds["split"] = "outer_test"
    test_preds.to_csv(fold_dir / "outer_test_predictions.csv", index=False)
    pd.DataFrame(history).to_csv(fold_dir / "history.csv", index=False)
    result = {
        "fold": fold,
        "best_epoch": best["epoch"],
        "val_roc_auc": best["val_roc_auc"],
        "val_threshold": best["val_threshold"],
        "test": test_metrics,
        "n_train": int(len(train_df)),
        "n_inner_val": int(len(val_df)),
        "n_outer_test": int(len(test_df)),
    }
    (fold_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = ["f1", "roc_auc", "accuracy", "balanced_accuracy", "pr_auc", "sensitivity", "specificity"]
    summary = {}
    for metric in metrics:
        values = np.array([fold["test"][metric] for fold in results], dtype=float)
        summary[f"{metric}_mean"] = float(values.mean())
        summary[f"{metric}_std"] = float(values.std(ddof=0))
    return {"summary": summary, "folds": results}


def main() -> None:
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    set_seed(int(config.get("seed", 42)))
    maybe_build_manifest(config, force=args.rebuild_manifest)
    manifest = pd.read_csv(config["data"]["manifest_csv"], low_memory=False)
    split_df = pd.read_csv(config["data"]["split_csv"], low_memory=False)
    device = torch.device(config.get("device", "cuda") if torch.cuda.is_available() else "cpu")
    Path(config["output_dir"]).mkdir(parents=True, exist_ok=True)
    folds = sorted(split_df["fold"].unique().astype(int).tolist())
    if args.folds:
        requested = {int(x) for x in args.folds.split(",") if x.strip()}
        folds = [fold for fold in folds if fold in requested]
    print(json.dumps({"device": str(device), "folds": folds, "n_manifest": len(manifest)}, ensure_ascii=False), flush=True)
    results = [run_fold(config, fold, manifest, split_df, device) for fold in folds]
    output = summarize(results)
    out_path = Path(config["output_dir"]) / "summary.json"
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2), flush=True)


if __name__ == "__main__":
    main()
