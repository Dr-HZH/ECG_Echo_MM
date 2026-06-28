#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from torch import amp, nn
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets.echo_dicom_dataset import EchoDicomDataset
from models import EchoBinaryClassifier, PanEchoIndustrialClassifier, PanEchoRefinementClassifier, build_video_backbone
from utils.metrics import compute_binary_metrics, find_best_threshold


class FocalWithLogitsLoss(nn.Module):
    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        pos_weight: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.register_buffer("pos_weight", pos_weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = torch.nn.functional.binary_cross_entropy_with_logits(
            logits,
            targets,
            reduction="none",
            pos_weight=self.pos_weight,
        )
        probs = torch.sigmoid(logits)
        pt = probs * targets + (1.0 - probs) * (1.0 - targets)
        alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
        return (alpha_t * (1.0 - pt).pow(self.gamma) * bce).mean()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Echo binary classifier with nested CV split.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--fold", type=int, default=None, help="Run a single fold only.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--limit_train_rows", type=int, default=None)
    parser.add_argument("--limit_val_rows", type=int, default=None)
    parser.add_argument("--limit_test_rows", type=int, default=None)
    parser.add_argument("--epochs_override", type=int, default=None)
    parser.add_argument("--batch_size_override", type=int, default=None)
    parser.add_argument("--output_dir_override", default=None)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def collate_echo_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    out["echo"] = torch.stack([item["echo"] for item in batch], dim=0)
    if "label" in batch[0]:
        out["label"] = torch.stack([item["label"] for item in batch], dim=0)
    for key in batch[0].keys():
        if key in {"echo", "label"}:
            continue
        out[key] = [item.get(key) for item in batch]
    return out


def make_dataloader(
    dataframe: pd.DataFrame,
    config: dict[str, Any],
    split: str,
    shuffle: bool,
) -> DataLoader:
    data_cfg = config["data"]
    aug_cfg = config.get("augment", {})
    dataset = EchoDicomDataset(
        csv_path=data_cfg["csv_path"],
        dataframe=dataframe,
        echo_path_col=data_cfg["path_col"],
        label_col=data_cfg["label_col"],
        patient_id_col=data_cfg["patient_id_col"],
        split="train" if split == "train" else "val",
        num_frames=data_cfg["num_frames"],
        image_size=data_cfg["image_size"],
        target_channels=data_cfg.get("target_channels", 3),
        input_format=data_cfg.get("input_format", "dicom"),
        data_root=data_cfg["data_root"],
        h5_path_col=data_cfg.get("path_col", data_cfg.get("h5_path_col", "h5_path")),
        h5_key_col=data_cfg.get("h5_key_col", "h5_key"),
        h5_root=data_cfg.get("h5_root", data_cfg.get("data_root", ".")),
        cache_dir=data_cfg.get("cache_dir"),
        cache_image_size=data_cfg.get("cache_image_size", data_cfg.get("image_size")),
        crop_scale_range=tuple(aug_cfg.get("crop_scale_range", (1.0, 1.0))),
        brightness_jitter=float(aug_cfg.get("brightness_jitter", 0.0)),
        contrast_jitter=float(aug_cfg.get("contrast_jitter", 0.0)),
        translate_fraction=float(aug_cfg.get("translate_fraction", 0.0)),
        normalize_mean=data_cfg.get("normalize_mean"),
        normalize_std=data_cfg.get("normalize_std"),
    )
    return DataLoader(
        dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=shuffle,
        num_workers=int(config["training"].get("num_workers", 0)),
        pin_memory=bool(config["training"].get("pin_memory", False)),
        collate_fn=collate_echo_batch,
    )


def make_model(config: dict[str, Any]) -> EchoBinaryClassifier:
    backbone_spec = build_video_backbone(config["model"])
    classifier = config["model"].get("classifier", "linear")
    if classifier == "panecho_refinement":
        if config["model"]["name"] != "panecho":
            raise ValueError("classifier=panecho_refinement requires model.name=panecho.")
        refine_cfg = config["model"].get("refinement", {})
        model = PanEchoRefinementClassifier(
            panecho_backbone=backbone_spec.backbone,
            feature_dim=backbone_spec.feature_dim,
            hidden_dim=int(refine_cfg.get("hidden_dim", 256)),
            dropout=float(config["model"].get("dropout", 0.2)),
            temporal_kernel_size=int(refine_cfg.get("temporal_kernel_size", 3)),
            aux_tasks=list(refine_cfg.get("aux_tasks", [])),
        )
    elif classifier == "panecho_industrial":
        if config["model"]["name"] != "panecho":
            raise ValueError("classifier=panecho_industrial requires model.name=panecho.")
        model = PanEchoIndustrialClassifier(
            panecho_backbone=backbone_spec.backbone,
            feature_dim=backbone_spec.feature_dim,
            dropout=float(config["model"].get("dropout", 0.2)),
            last_frame_weight=float(config["model"].get("last_frame_weight", 0.0)),
        )
    else:
        model = EchoBinaryClassifier(
            backbone=backbone_spec.backbone,
            feature_dim=backbone_spec.feature_dim,
            dropout=float(config["model"].get("dropout", 0.2)),
        )
    model.configure_freezing(
        freeze_mode=config["model"].get("freeze_mode", "frozen"),
        unfreeze_patterns=config["model"].get("unfreeze_patterns", []),
    )
    model.pretrained_info = backbone_spec.pretrained_info
    return model


def build_optimizer(model: nn.Module, config: dict[str, Any]) -> torch.optim.Optimizer:
    train_cfg = config["training"]
    lr_backbone = float(train_cfg.get("lr_backbone", 1e-5))
    lr_head = float(train_cfg.get("lr_head", 1e-3))
    weight_decay = float(train_cfg.get("weight_decay", 1e-4))

    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
    head_modules = ["temporal", "attention_pool", "feature_gate", "pool", "head", "aux_heads"]
    head_params = []
    for module_name in head_modules:
        module = getattr(model, module_name, None)
        if module is not None:
            head_params.extend([p for p in module.parameters() if p.requires_grad])
    param_groups = []
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": lr_backbone})
    if head_params:
        param_groups.append({"params": head_params, "lr": lr_head})
    return torch.optim.AdamW(param_groups, weight_decay=weight_decay)


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    config: dict[str, Any],
    steps_per_epoch: int,
) -> torch.optim.lr_scheduler.LRScheduler | None:
    train_cfg = config["training"]
    scheduler_name = train_cfg.get("scheduler", "none")
    if scheduler_name in (None, "none"):
        return None
    if scheduler_name != "cosine":
        raise ValueError(f"Unsupported scheduler: {scheduler_name}")

    total_steps = max(1, steps_per_epoch * int(train_cfg["epochs"]))
    warmup_ratio = float(train_cfg.get("warmup_ratio", train_cfg.get("warmup", 0.0)))
    warmup_steps = int(total_steps * warmup_ratio)

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def maybe_load_init_checkpoint(model: nn.Module, config: dict[str, Any]) -> None:
    ckpt_path = config["training"].get("init_checkpoint")
    if not ckpt_path:
        return
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    state_dict = payload.get("model_state_dict", payload)
    model.load_state_dict(state_dict, strict=False)


def make_loss(train_df: pd.DataFrame, config: dict[str, Any], device: str) -> nn.Module:
    use_pos_weight = bool(config["training"].get("use_pos_weight", False))
    loss_name = str(config["training"].get("loss", "bce")).lower()
    pos_weight_tensor = None
    labels = pd.to_numeric(train_df[config["data"]["label_col"]], errors="coerce").astype(float)
    if use_pos_weight:
        pos = float((labels == 1).sum())
        neg = float((labels == 0).sum())
        if pos <= 0 or neg <= 0:
            raise ValueError("Cannot compute pos_weight without both classes in train split.")
        pos_weight = neg / pos
        lower = float(config["training"].get("pos_weight_min", 0.5))
        upper = float(config["training"].get("pos_weight_max", 2.0))
        pos_weight = min(max(pos_weight, lower), upper)
        pos_weight_tensor = torch.tensor([pos_weight], device=device)

    if loss_name == "bce":
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    if loss_name == "focal":
        return FocalWithLogitsLoss(
            alpha=float(config["training"].get("focal_alpha", 0.25)),
            gamma=float(config["training"].get("focal_gamma", 2.0)),
            pos_weight=pos_weight_tensor,
        )
    raise ValueError(f"Unsupported loss: {loss_name}")


def predict_logits(model: nn.Module, x: torch.Tensor, tta_shifts: list[int] | None = None) -> torch.Tensor:
    if not tta_shifts:
        return model(x)
    logits = []
    for shift in tta_shifts:
        if shift == 0:
            x_view = x
        else:
            x_view = torch.roll(x, shifts=int(shift), dims=2)
        logits.append(model(x_view))
    return torch.stack(logits, dim=0).mean(dim=0)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: str,
    use_amp: bool,
    scaler: amp.GradScaler | None,
    gradient_accumulation_steps: int = 1,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    tta_shifts: list[int] | None = None,
) -> dict[str, Any]:
    training = optimizer is not None
    model.train(training)
    losses = []
    y_true = []
    y_prob = []

    if training:
        optimizer.zero_grad(set_to_none=True)

    for step_idx, batch in enumerate(loader, start=1):
        x = batch["echo"].to(device)
        y = batch["label"].to(device).float()

        with amp.autocast(device_type="cuda", enabled=use_amp):
            logits = predict_logits(model, x, tta_shifts=None if training else tta_shifts)
            loss = criterion(logits, y)

        if training:
            assert scaler is not None
            scaled_loss = loss / gradient_accumulation_steps
            scaler.scale(scaled_loss).backward()
            if step_idx % gradient_accumulation_steps == 0 or step_idx == len(loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                if scheduler is not None:
                    scheduler.step()

        losses.append(float(loss.detach().cpu()))
        y_true.append(y.detach().cpu().numpy())
        y_prob.append(torch.sigmoid(logits.detach()).cpu().numpy())

    y_true_arr = np.concatenate(y_true)
    y_prob_arr = np.concatenate(y_prob)
    threshold = 0.5
    if np.unique(y_true_arr).shape[0] < 2:
        metrics = {
            "loss": float(np.mean(losses)),
            "y_true": y_true_arr,
            "y_prob": y_prob_arr,
        }
        return metrics

    threshold = find_best_threshold(y_true_arr, y_prob_arr, strategy="youden")
    binary_metrics = compute_binary_metrics(y_true_arr, y_prob_arr, threshold=threshold)
    return {
        "loss": float(np.mean(losses)),
        "y_true": y_true_arr,
        "y_prob": y_prob_arr,
        "threshold": threshold,
        **binary_metrics.__dict__,
    }


def subset_df(df: pd.DataFrame, limit: int | None) -> pd.DataFrame:
    if limit is None or len(df) <= limit:
        return df
    return df.iloc[:limit].reset_index(drop=True)


def train_fold(
    manifest_df: pd.DataFrame,
    split_df: pd.DataFrame,
    config: dict[str, Any],
    fold: int,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], pd.DataFrame]:
    data_cfg = config["data"]
    label_col = data_cfg["label_col"]
    id_col = data_cfg["id_col"]
    merged = manifest_df.merge(
        split_df[split_df["fold"] == fold],
        on=[id_col, data_cfg["patient_id_col"]],
        how="inner",
        suffixes=("", "_split"),
    )

    train_df = subset_df(merged[merged["split"] == "train"].copy(), args.limit_train_rows)
    val_df = subset_df(merged[merged["split"] == "inner_val"].copy(), args.limit_val_rows)
    test_df = subset_df(merged[merged["split"] == "outer_test"].copy(), args.limit_test_rows)
    if train_df.empty or val_df.empty or test_df.empty:
        raise ValueError(f"Fold {fold} has empty split after filtering.")
    print(
        json.dumps(
            {
                "event": "fold_start",
                "fold": fold,
                "n_train": int(len(train_df)),
                "n_inner_val": int(len(val_df)),
                "n_outer_test": int(len(test_df)),
                "train_positive_rate": float(pd.to_numeric(train_df[label_col], errors="coerce").mean()),
                "val_positive_rate": float(pd.to_numeric(val_df[label_col], errors="coerce").mean()),
                "test_positive_rate": float(pd.to_numeric(test_df[label_col], errors="coerce").mean()),
            }
        ),
        flush=True,
    )

    train_loader = make_dataloader(train_df, config=config, split="train", shuffle=True)
    val_loader = make_dataloader(val_df, config=config, split="val", shuffle=False)
    test_loader = make_dataloader(test_df, config=config, split="test", shuffle=False)

    device = args.device
    model = make_model(config).to(device)
    print(
        json.dumps(
            {
                "event": "pretrained_audit",
                "model": config["model"]["name"],
                **model.pretrained_info,
            }
        ),
        flush=True,
    )
    maybe_load_init_checkpoint(model, config)
    optimizer = build_optimizer(model, config)
    grad_accum = max(1, int(config["training"].get("gradient_accumulation_steps", 1)))
    steps_per_epoch = max(1, math.ceil(len(train_loader) / grad_accum))
    scheduler = build_scheduler(optimizer, config=config, steps_per_epoch=steps_per_epoch)
    criterion = make_loss(train_df, config=config, device=device)
    use_amp = bool(config["training"].get("amp", False)) and device.startswith("cuda")
    scaler = amp.GradScaler("cuda", enabled=use_amp)

    output_root = Path(config["experiment"]["output_dir"])
    fold_dir = output_root / f"fold_{fold:02d}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    best_val_auroc = -math.inf
    best_epoch = -1
    best_threshold = 0.5
    patience = int(config["checkpoint"].get("patience", 5))
    bad_epochs = 0
    checkpoint_path = fold_dir / "best.ckpt"

    for epoch in range(1, int(config["training"]["epochs"]) + 1):
        epoch_start = time.time()
        train_metrics = run_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            use_amp,
            scaler,
            gradient_accumulation_steps=grad_accum,
            scheduler=scheduler,
        )
        val_metrics = run_epoch(model, val_loader, criterion, None, device, use_amp=False, scaler=None)
        if "auroc" not in val_metrics:
            raise ValueError(f"Fold {fold} validation split has one class only; nested split is invalid.")

        epoch_log = {
            "event": "epoch_end",
            "fold": fold,
            "epoch": epoch,
            "train_loss": float(train_metrics["loss"]),
            "val_loss": float(val_metrics["loss"]),
            "val_auroc": float(val_metrics["auroc"]),
            "val_auprc": float(val_metrics["auprc"]),
            "val_f1": float(val_metrics["f1"]),
            "val_threshold": float(val_metrics["threshold"]),
            "epoch_seconds": round(time.time() - epoch_start, 3),
        }

        if val_metrics["auroc"] > best_val_auroc:
            best_val_auroc = float(val_metrics["auroc"])
            best_epoch = epoch
            best_threshold = float(val_metrics["threshold"])
            bad_epochs = 0
            epoch_log["is_best"] = True
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
                    "epoch": epoch,
                    "fold": fold,
                    "best_val_auroc": best_val_auroc,
                    "config": config,
                    "pretrained_info": model.pretrained_info,
                    "val_metrics": {k: v for k, v in val_metrics.items() if k not in {"y_true", "y_prob"}},
                    "train_metrics": {k: v for k, v in train_metrics.items() if k not in {"y_true", "y_prob"}},
                },
                checkpoint_path,
            )
        else:
            bad_epochs += 1
            epoch_log["is_best"] = False
            epoch_log["bad_epochs"] = bad_epochs
        print(json.dumps(epoch_log), flush=True)
        if bad_epochs >= patience:
            print(
                json.dumps(
                    {
                        "event": "early_stop",
                        "fold": fold,
                        "epoch": epoch,
                        "best_epoch": best_epoch,
                        "best_val_auroc": best_val_auroc,
                    }
                ),
                flush=True,
            )
            break

    payload = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(payload["model_state_dict"])
    eval_cfg = config.get("evaluation", {})
    tta_shifts = [int(x) for x in eval_cfg.get("tta_shifts", [])]
    if not bool(eval_cfg.get("use_tta", False)):
        tta_shifts = []

    val_metrics = run_epoch(
        model,
        val_loader,
        criterion,
        None,
        device,
        use_amp=False,
        scaler=None,
        tta_shifts=tta_shifts,
    )
    test_metrics_raw = run_epoch(
        model,
        test_loader,
        criterion,
        None,
        device,
        use_amp=False,
        scaler=None,
        tta_shifts=tta_shifts,
    )
    eval_threshold = float(val_metrics.get("threshold", best_threshold))
    test_metrics = compute_binary_metrics(
        test_metrics_raw["y_true"],
        test_metrics_raw["y_prob"],
        threshold=eval_threshold,
    )

    test_preds = test_df[[id_col, data_cfg["patient_id_col"], label_col, data_cfg["path_col"]]].copy()
    test_preds["probability"] = test_metrics_raw["y_prob"]
    test_preds["predicted_label"] = (test_preds["probability"] >= eval_threshold).astype(int)
    test_preds["fold"] = fold
    test_preds["checkpoint_path"] = str(checkpoint_path)

    fold_summary = {
        "fold": fold,
        "best_epoch": best_epoch,
        "best_val_auroc": best_val_auroc,
        "best_threshold": best_threshold,
        "eval_threshold": eval_threshold,
        "n_train": int(len(train_df)),
        "n_inner_val": int(len(val_df)),
        "n_outer_test": int(len(test_df)),
        "outer_test_metrics": test_metrics.__dict__,
        "pretrained_info": model.pretrained_info,
        "checkpoint_path": str(checkpoint_path),
    }
    (fold_dir / "fold_summary.json").write_text(json.dumps(fold_summary, indent=2), encoding="utf-8")
    return fold_summary, test_preds


def main() -> None:
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if args.epochs_override is not None:
        config["training"]["epochs"] = int(args.epochs_override)
    if args.batch_size_override is not None:
        config["training"]["batch_size"] = int(args.batch_size_override)
    if args.output_dir_override is not None:
        config["experiment"]["output_dir"] = args.output_dir_override

    seed_everything(int(config["experiment"].get("seed", 42)))
    output_dir = Path(config["experiment"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_df = pd.read_csv(config["data"]["csv_path"], low_memory=False)
    if "eligible_for_training" in manifest_df.columns:
        manifest_df = manifest_df[manifest_df["eligible_for_training"] == 1].copy()
    split_df = pd.read_csv(config["cross_validation"]["split_file"], low_memory=False)

    folds = [args.fold] if args.fold is not None else sorted(split_df["fold"].unique().tolist())
    fold_summaries = []
    oof_parts = []
    for fold in folds:
        summary, preds = train_fold(manifest_df=manifest_df, split_df=split_df, config=config, fold=int(fold), args=args)
        fold_summaries.append(summary)
        oof_parts.append(preds)

    oof = pd.concat(oof_parts, ignore_index=True)
    oof_path = output_dir / "oof_predictions.csv"
    oof.to_csv(oof_path, index=False)

    aggregate = {
        "folds": fold_summaries,
        "oof_path": str(oof_path),
    }
    (output_dir / "run_summary.json").write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
