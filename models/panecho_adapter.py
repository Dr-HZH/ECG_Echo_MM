from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import torch
import torchvision
import pandas as pd
import numpy as np
from torch import nn


@dataclass
class PanEchoTask:
    task_name: str
    task_type: str
    class_names: np.ndarray
    mean: float = float("nan")


def _load_official_models_module(repo_path: Path) -> ModuleType:
    models_path = repo_path / "src" / "models.py"
    if not models_path.is_file():
        raise FileNotFoundError(f"PanEcho model source not found: {models_path}")

    spec = importlib.util.spec_from_file_location("_official_panecho_models", models_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import PanEcho model source: {models_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _build_official_backbone(repo_path: Path, clip_len: int) -> nn.Module:
    module = _load_official_models_module(repo_path)

    # The full PanEcho checkpoint contains the ConvNeXt weights. Avoid a second
    # ImageNet download while preserving the official FrameTransformer class.
    original_convnext_tiny = torchvision.models.convnext_tiny

    def convnext_tiny_without_download(*args: Any, **kwargs: Any) -> nn.Module:
        kwargs["weights"] = None
        return original_convnext_tiny(*args, **kwargs)

    torchvision.models.convnext_tiny = convnext_tiny_without_download
    try:
        backbone = module.FrameTransformer(
            arch="convnext_tiny",
            n_heads=8,
            n_layers=4,
            transformer_dropout=0.0,
            pooling="mean",
            clip_len=clip_len,
        )
    finally:
        torchvision.models.convnext_tiny = original_convnext_tiny
    return backbone


def _extract_encoder_weights(checkpoint: dict[str, Any]) -> dict[str, torch.Tensor]:
    if "weights" not in checkpoint or not isinstance(checkpoint["weights"], dict):
        raise RuntimeError("PanEcho checkpoint must contain a `weights` state dict.")

    full_state = checkpoint["weights"]
    encoder_state = {
        key.removeprefix("encoder."): value
        for key, value in full_state.items()
        if key.startswith("encoder.")
    }
    if not encoder_state:
        raise RuntimeError("PanEcho checkpoint contains no `encoder.*` weights.")

    encoder_state.pop("time_encoder.pe", None)
    return encoder_state


def build_local_panecho_backbone(
    repo_path: str | Path,
    checkpoint_path: str | Path | None = None,
    clip_len: int = 16,
) -> tuple[nn.Module, dict[str, Any]]:
    repo_path = Path(repo_path)
    if not repo_path.is_dir():
        raise FileNotFoundError(f"PanEcho repository not found: {repo_path}")

    backbone = _build_official_backbone(repo_path=repo_path, clip_len=clip_len)
    if checkpoint_path is None:
        first_conv = backbone.encoder.model.features[0][0].weight
        audit = {
            "pretrained_loaded": False,
            "pretrained_source": "random_structure_only",
            "checkpoint_path": None,
            "repo_path": str(repo_path),
            "feature_dim": int(backbone.encoder.n_features),
            "first_conv_mean": float(first_conv.detach().mean().cpu()),
            "missing_keys": [],
            "unexpected_keys": [],
        }
        return backbone, audit

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"PanEcho checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise RuntimeError(f"Unsupported PanEcho checkpoint type: {type(checkpoint)}")

    encoder_state = _extract_encoder_weights(checkpoint)
    incompatible = backbone.load_state_dict(encoder_state, strict=False)
    allowed_missing = {"time_encoder.pe"}
    missing = set(incompatible.missing_keys)
    unexpected = set(incompatible.unexpected_keys)
    if missing != allowed_missing or unexpected:
        raise RuntimeError(
            "PanEcho pretrained weights did not match the official backbone: "
            f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
        )

    first_conv = backbone.encoder.model.features[0][0].weight
    audit = {
        "pretrained_loaded": True,
        "pretrained_source": "local",
        "checkpoint_path": str(checkpoint_path),
        "repo_path": str(repo_path),
        "feature_dim": int(backbone.encoder.n_features),
        "first_conv_mean": float(first_conv.detach().mean().cpu()),
        "missing_keys": sorted(missing),
        "unexpected_keys": sorted(unexpected),
    }
    return backbone, audit


def build_local_panecho_model(
    repo_path: str | Path,
    checkpoint_path: str | Path,
    clip_len: int = 16,
) -> nn.Module:
    """Build the official PanEcho model with all native heads, fully offline."""
    repo_path = Path(repo_path)
    checkpoint_path = Path(checkpoint_path)
    task_path = repo_path / "content" / "tasks.pkl"
    if not task_path.is_file() or not checkpoint_path.is_file():
        raise FileNotFoundError("PanEcho tasks.pkl and checkpoint are required.")

    module = _load_official_models_module(repo_path)
    task_dict = pd.read_pickle(task_path)
    tasks = [
        PanEchoTask(name, spec["task_type"], np.asarray(spec["class_names"]), spec["mean"])
        for name, spec in task_dict.items()
    ]
    encoder = _build_official_backbone(repo_path, clip_len)
    model = module.MultiTaskModel(encoder, encoder.encoder.n_features, tasks, 0.25, True)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("weights") if isinstance(checkpoint, dict) else None
    if not isinstance(state_dict, dict):
        raise RuntimeError("PanEcho checkpoint must contain a `weights` state dict.")
    state_dict = dict(state_dict)
    state_dict.pop("encoder.time_encoder.pe", None)
    incompatible = model.load_state_dict(state_dict, strict=False)
    if set(incompatible.missing_keys) != {"encoder.time_encoder.pe"} or incompatible.unexpected_keys:
        raise RuntimeError(
            f"PanEcho full-model mismatch: missing={incompatible.missing_keys}, "
            f"unexpected={incompatible.unexpected_keys}"
        )
    return model
