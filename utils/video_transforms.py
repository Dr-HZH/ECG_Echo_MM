from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F


def minmax_scale_video(tensor: torch.Tensor) -> torch.Tensor:
    min_val = tensor.amin()
    max_val = tensor.amax()
    if torch.isfinite(min_val) and torch.isfinite(max_val) and max_val > min_val:
        return (tensor - min_val) / (max_val - min_val)
    return tensor


def _crop_video(
    tensor: torch.Tensor,
    top: int,
    left: int,
    crop_h: int,
    crop_w: int,
) -> torch.Tensor:
    return tensor[:, :, top : top + crop_h, left : left + crop_w]


def crop_video(
    tensor: torch.Tensor,
    split: str,
    crop_scale_range: tuple[float, float] | None,
    rng: np.random.Generator | None = None,
) -> torch.Tensor:
    if crop_scale_range is None:
        return tensor

    c, t, h, w = tensor.shape
    if rng is None:
        rng = np.random.default_rng()
    min_scale, max_scale = crop_scale_range
    scale = max_scale if split != "train" else float(rng.uniform(min_scale, max_scale))
    crop_h = max(1, min(h, int(round(h * scale))))
    crop_w = max(1, min(w, int(round(w * scale))))

    if crop_h == h and crop_w == w:
        return tensor

    max_top = h - crop_h
    max_left = w - crop_w
    if split == "train":
        top = int(rng.integers(0, max_top + 1)) if max_top > 0 else 0
        left = int(rng.integers(0, max_left + 1)) if max_left > 0 else 0
    else:
        top = max_top // 2
        left = max_left // 2
    return _crop_video(tensor, top=top, left=left, crop_h=crop_h, crop_w=crop_w)


def translate_video(
    tensor: torch.Tensor,
    split: str,
    max_translate_fraction: float = 0.0,
    rng: np.random.Generator | None = None,
) -> torch.Tensor:
    if max_translate_fraction <= 0:
        return tensor
    _, _, h, w = tensor.shape
    if rng is None:
        rng = np.random.default_rng()

    max_dy = int(round(h * max_translate_fraction))
    max_dx = int(round(w * max_translate_fraction))
    if split == "train":
        dy = int(rng.integers(-max_dy, max_dy + 1)) if max_dy > 0 else 0
        dx = int(rng.integers(-max_dx, max_dx + 1)) if max_dx > 0 else 0
    else:
        dy = dx = 0

    if dy == 0 and dx == 0:
        return tensor

    out = torch.roll(tensor, shifts=(dy, dx), dims=(-2, -1))
    if dy > 0:
        out[:, :, :dy, :] = 0
    elif dy < 0:
        out[:, :, dy:, :] = 0
    if dx > 0:
        out[:, :, :, :dx] = 0
    elif dx < 0:
        out[:, :, :, dx:] = 0
    return out


def jitter_brightness_contrast(
    tensor: torch.Tensor,
    split: str,
    brightness: float = 0.0,
    contrast: float = 0.0,
    rng: np.random.Generator | None = None,
) -> torch.Tensor:
    if split != "train":
        return tensor
    if rng is None:
        rng = np.random.default_rng()

    out = tensor
    if brightness > 0:
        delta = float(rng.uniform(-brightness, brightness))
        out = out + delta
    if contrast > 0:
        factor = float(rng.uniform(max(0.0, 1.0 - contrast), 1.0 + contrast))
        mean = out.mean(dim=(-2, -1), keepdim=True)
        out = (out - mean) * factor + mean
    return out.clamp(0.0, 1.0)


def resize_video(
    tensor: torch.Tensor,
    image_size: int | tuple[int, int] | None,
) -> torch.Tensor:
    if image_size is None:
        return tensor
    if isinstance(image_size, int):
        target_h = target_w = image_size
    else:
        target_h, target_w = image_size

    if tensor.shape[-2:] == (target_h, target_w):
        return tensor

    video = tensor.permute(1, 0, 2, 3)
    video = F.interpolate(video, size=(target_h, target_w), mode="bilinear", align_corners=False)
    return video.permute(1, 0, 2, 3)


def normalize_video(
    tensor: torch.Tensor,
    mean: Sequence[float] | None,
    std: Sequence[float] | None,
) -> torch.Tensor:
    if mean is None or std is None:
        return tensor
    mean_tensor = torch.tensor(mean, dtype=tensor.dtype, device=tensor.device).view(-1, 1, 1, 1)
    std_tensor = torch.tensor(std, dtype=tensor.dtype, device=tensor.device).view(-1, 1, 1, 1)
    return (tensor - mean_tensor) / std_tensor
