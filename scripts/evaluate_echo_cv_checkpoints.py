#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from torch import amp

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.train_echo_cv import make_dataloader, make_model, predict_logits, seed_everything
from utils.metrics import compute_binary_metrics, find_best_threshold


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Echo CV checkpoints with TTA and temperature scaling.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint_root", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--temperatures", nargs="+", type=float, default=[1.0, 1.2, 1.5, 2.0])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def collect_probs(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: str,
    tta_shifts: list[int],
    temperature: float,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    y_true: list[np.ndarray] = []
    y_prob: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            x = batch["echo"].to(device)
            y = batch["label"].to(device).float()
            with amp.autocast(device_type="cuda", enabled=False):
                logits = predict_logits(model, x, tta_shifts=tta_shifts)
                probs = torch.sigmoid(logits / temperature)
            y_true.append(y.detach().cpu().numpy())
            y_prob.append(probs.detach().cpu().numpy())
    return np.concatenate(y_true), np.concatenate(y_prob)


def summarize_folds(folds: list[dict[str, Any]]) -> dict[str, float]:
    summary: dict[str, float] = {}
    for key in ["f1", "roc_auc", "accuracy", "balanced_accuracy", "pr_auc", "sensitivity", "specificity"]:
        values = np.asarray([fold[key] for fold in folds], dtype=float)
        summary[f"{key}_mean"] = float(values.mean())
        summary[f"{key}_std"] = float(values.std(ddof=1))
    return summary


def main() -> None:
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    seed_everything(int(config["experiment"].get("seed", 42)))

    manifest_df = pd.read_csv(config["data"]["csv_path"], low_memory=False)
    if "eligible_for_training" in manifest_df.columns:
        manifest_df = manifest_df[manifest_df["eligible_for_training"] == 1].copy()
    split_df = pd.read_csv(config["cross_validation"]["split_file"], low_memory=False)

    eval_cfg = config.get("evaluation", {})
    tta_shifts = [int(x) for x in eval_cfg.get("tta_shifts", [])] if eval_cfg.get("use_tta", False) else []
    data_cfg = config["data"]
    id_col = data_cfg["id_col"]
    patient_col = data_cfg["patient_id_col"]
    label_col = data_cfg["label_col"]
    checkpoint_root = Path(args.checkpoint_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, Any] = {}
    all_prediction_parts = []

    for temperature in args.temperatures:
        folds_out = []
        prediction_parts = []
        for fold in sorted(split_df["fold"].unique().tolist()):
            fold = int(fold)
            merged = manifest_df.merge(
                split_df[split_df["fold"] == fold],
                on=[id_col, patient_col],
                how="inner",
                suffixes=("", "_split"),
            )
            val_df = merged[merged["split"] == "inner_val"].copy()
            test_df = merged[merged["split"] == "outer_test"].copy()
            val_loader = make_dataloader(val_df, config=config, split="val", shuffle=False)
            test_loader = make_dataloader(test_df, config=config, split="test", shuffle=False)

            model = make_model(config).to(args.device)
            ckpt_path = checkpoint_root / f"fold_{fold:02d}" / "best.ckpt"
            payload = torch.load(ckpt_path, map_location=args.device, weights_only=True)
            model.load_state_dict(payload["model_state_dict"])

            y_val, p_val = collect_probs(model, val_loader, args.device, tta_shifts, temperature)
            threshold = find_best_threshold(y_val, p_val, strategy="youden")
            y_test, p_test = collect_probs(model, test_loader, args.device, tta_shifts, temperature)
            metrics = compute_binary_metrics(y_test, p_test, threshold=threshold)
            sensitivity = float(metrics.sensitivity)
            specificity = float(metrics.specificity)
            folds_out.append(
                {
                    "fold": fold,
                    "temperature": float(temperature),
                    "threshold": float(threshold),
                    "f1": float(metrics.f1),
                    "roc_auc": float(metrics.auroc),
                    "accuracy": float(metrics.accuracy),
                    "balanced_accuracy": float((sensitivity + specificity) / 2.0),
                    "pr_auc": float(metrics.auprc),
                    "sensitivity": sensitivity,
                    "specificity": specificity,
                    "checkpoint_path": str(ckpt_path),
                }
            )
            pred_df = test_df[[id_col, patient_col, label_col, data_cfg["path_col"]]].copy()
            pred_df["probability"] = p_test
            pred_df["predicted_label"] = (pred_df["probability"] >= threshold).astype(int)
            pred_df["fold"] = fold
            pred_df["temperature"] = float(temperature)
            prediction_parts.append(pred_df)

        summary = summarize_folds(folds_out)
        key = f"T={temperature:g}"
        results[key] = {"folds": folds_out, "summary": summary}
        temp_predictions = pd.concat(prediction_parts, ignore_index=True)
        temp_predictions.to_csv(output_dir / f"predictions_temperature_{temperature:g}.csv", index=False)
        all_prediction_parts.append(temp_predictions)
        print(json.dumps({"temperature": temperature, "summary": summary}, ensure_ascii=False), flush=True)

    best_key = max(results, key=lambda k: results[k]["summary"]["roc_auc_mean"])
    final = {
        "best_by_roc_auc": best_key,
        "results": results,
        "note": "Temperature scaling is monotonic per fold; AUROC changes only due to checkpoint/TTA prediction collection, not thresholding.",
    }
    (output_dir / "temperature_sweep_summary.json").write_text(
        json.dumps(final, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
