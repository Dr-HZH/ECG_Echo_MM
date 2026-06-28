from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from models.panecho_adapter import build_local_panecho_backbone


@dataclass
class BackboneSpec:
    backbone: nn.Module
    feature_dim: int
    pretrained_info: dict[str, Any]


def _pool_features(output: Any) -> torch.Tensor:
    if isinstance(output, tuple):
        output = output[0]
    if not isinstance(output, torch.Tensor):
        raise TypeError(f"Backbone forward returned unsupported type: {type(output)}")
    if output.ndim == 5:
        return output.mean(dim=(-1, -2, -3))
    if output.ndim == 4:
        return output.mean(dim=(-1, -2))
    if output.ndim == 3:
        return output.mean(dim=-1)
    if output.ndim == 2:
        return output
    raise ValueError(f"Unsupported backbone output shape: {tuple(output.shape)}")


class GenericVideoBackboneAdapter(nn.Module):
    def __init__(
        self,
        module: nn.Module,
        forward_method: str = "forward_features",
    ) -> None:
        super().__init__()
        self.module = module
        self.forward_method = forward_method

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self.module, self.forward_method):
            output = getattr(self.module, self.forward_method)(x)
        else:
            output = self.module(x)
        return _pool_features(output)


def build_r2plus1d_18(
    pretrained: bool = True,
    pretrained_source: str = "torchvision",
    checkpoint_path: str | None = None,
) -> BackboneSpec:
    try:
        from torchvision.models.video import R2Plus1D_18_Weights, r2plus1d_18  # type: ignore
    except Exception as exc:
        raise RuntimeError("torchvision with video models is required for r2plus1d_18.") from exc

    torch_home = Path(__file__).resolve().parents[1] / ".cache" / "torch"
    torch_home.mkdir(parents=True, exist_ok=True)
    torch.hub.set_dir(str(torch_home))
    resolved_checkpoint: str | None = None
    if not pretrained:
        if checkpoint_path:
            raise ValueError("R2Plus1D checkpoint_path is set while pretrained=false.")
        model = r2plus1d_18(weights=None)
        pretrained_loaded = False
        resolved_source = "random"
    elif pretrained_source == "torchvision":
        model = r2plus1d_18(weights=R2Plus1D_18_Weights.DEFAULT)
        pretrained_loaded = True
        resolved_source = "torchvision"
        resolved_checkpoint = R2Plus1D_18_Weights.DEFAULT.url
    elif pretrained_source == "local":
        if not checkpoint_path:
            raise ValueError("R2Plus1D pretrained_source=local requires checkpoint_path.")
        checkpoint = Path(checkpoint_path)
        if not checkpoint.is_file():
            raise FileNotFoundError(f"R2Plus1D pretrained checkpoint not found: {checkpoint}")

        # Load into the untouched 400-class architecture so every official key
        # must match before the classifier is removed.
        model = r2plus1d_18(weights=None)
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        state_dict = payload.get("state_dict", payload) if isinstance(payload, dict) else payload
        model.load_state_dict(state_dict, strict=True)
        pretrained_loaded = True
        resolved_source = "local"
        resolved_checkpoint = str(checkpoint)
    else:
        raise ValueError(f"Unsupported R2Plus1D pretrained_source: {pretrained_source}")

    first_conv_mean = float(model.stem[0].weight.detach().mean().cpu())
    feature_dim = int(model.fc.in_features)
    model.fc = nn.Identity()
    return BackboneSpec(
        backbone=model,
        feature_dim=feature_dim,
        pretrained_info={
            "pretrained_loaded": pretrained_loaded,
            "pretrained_source": resolved_source,
            "checkpoint_path": resolved_checkpoint,
            "feature_dim": feature_dim,
            "first_conv_mean": first_conv_mean,
            "missing_keys": [],
            "unexpected_keys": [],
        },
    )


def build_panecho(model_cfg: dict[str, Any]) -> BackboneSpec:
    pretrained = bool(model_cfg.get("pretrained", True))
    if model_cfg.get("pretrained_source", "local") != "local":
        raise ValueError("PanEcho currently supports only pretrained_source=local.")

    panecho_cfg = model_cfg.get("panecho", {})
    checkpoint_path = model_cfg.get("checkpoint_path") or panecho_cfg.get("checkpoint_path")
    repo_path = panecho_cfg.get("repo_path")
    if not repo_path:
        raise ValueError("PanEcho requires panecho.repo_path.")
    if pretrained and not checkpoint_path:
        raise ValueError("PanEcho pretrained=true requires checkpoint_path.")

    module, audit = build_local_panecho_backbone(
        repo_path=repo_path,
        checkpoint_path=checkpoint_path if pretrained else None,
        clip_len=int(panecho_cfg.get("clip_len", 16)),
    )
    feature_dim = int(audit["feature_dim"])
    return BackboneSpec(backbone=module, feature_dim=feature_dim, pretrained_info=audit)


def build_video_backbone(model_cfg: dict[str, Any]) -> BackboneSpec:
    name = model_cfg["name"]
    if name == "r2plus1d_18":
        return build_r2plus1d_18(
            pretrained=bool(model_cfg.get("pretrained", True)),
            pretrained_source=model_cfg.get("pretrained_source", "torchvision"),
            checkpoint_path=model_cfg.get("checkpoint_path"),
        )
    if name == "panecho":
        return build_panecho(model_cfg)
    raise ValueError(f"Unsupported model name: {name}")
