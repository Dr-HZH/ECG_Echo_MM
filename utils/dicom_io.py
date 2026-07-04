from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pydicom


class DicomReadError(RuntimeError):
    """Raised when a DICOM file cannot be decoded into pixel data."""


@dataclass
class DicomReadResult:
    path: str
    array: np.ndarray
    original_shape: tuple[int, ...]
    original_dtype: str
    normalized_shape: tuple[int, int, int, int]
    transfer_syntax_uid: str | None
    photometric_interpretation: str | None
    samples_per_pixel: int | None
    number_of_frames: int
    channel_count: int
    frame_count: int
    height: int
    width: int
    decoding_plugin: str


def get_transfer_syntax_uid(dataset: pydicom.dataset.FileDataset) -> str | None:
    file_meta = getattr(dataset, "file_meta", None)
    if file_meta is None:
        return None
    uid = getattr(file_meta, "TransferSyntaxUID", None)
    return str(uid) if uid is not None else None


def _maybe_invert_monochrome1(
    array: np.ndarray,
    photometric_interpretation: str | None,
) -> np.ndarray:
    if photometric_interpretation != "MONOCHROME1":
        return array

    arr_min = array.min()
    arr_max = array.max()
    inverted = arr_max + arr_min - array
    if np.issubdtype(array.dtype, np.integer):
        return inverted.astype(array.dtype, copy=False)
    return inverted


def _normalize_pixel_array(
    array: np.ndarray,
    dataset: pydicom.dataset.FileDataset,
) -> np.ndarray:
    arr = np.asarray(array)
    frames = int(getattr(dataset, "NumberOfFrames", 1) or 1)
    spp = int(getattr(dataset, "SamplesPerPixel", 1) or 1)

    if arr.ndim == 2:
        out = arr[None, None, :, :]
    elif arr.ndim == 3:
        if frames > 1 and arr.shape[0] == frames:
            out = arr[None, :, :, :]
        elif spp > 1 and arr.shape[-1] == spp:
            out = np.transpose(arr, (2, 0, 1))[..., None]
            out = np.transpose(out, (0, 3, 1, 2))
        elif arr.shape[-1] in (1, 3, 4):
            out = np.transpose(arr, (2, 0, 1))[..., None]
            out = np.transpose(out, (0, 3, 1, 2))
        elif arr.shape[0] <= 4:
            out = arr[:, None, :, :]
        else:
            out = arr[None, :, :, :]
    elif arr.ndim == 4:
        if spp > 1 and arr.shape[-1] == spp:
            out = np.transpose(arr, (3, 0, 1, 2))
        elif arr.shape[0] <= 4 and arr.shape[1] > 4:
            out = arr
        elif arr.shape[-1] <= 4 and arr.shape[0] > 4:
            out = np.transpose(arr, (3, 0, 1, 2))
        elif arr.shape[1] <= 4 and arr.shape[0] > 4:
            out = np.transpose(arr, (1, 0, 2, 3))
        else:
            raise DicomReadError(f"Unsupported 4D pixel array shape: {arr.shape}")
    else:
        raise DicomReadError(f"Unsupported pixel array rank: {arr.ndim}")

    if out.ndim != 4:
        raise DicomReadError(f"Failed to normalize pixel array shape: {arr.shape} -> {out.shape}")
    return out


def _codec_hint(
    dataset: pydicom.dataset.FileDataset,
    exc: Exception,
) -> str:
    uid = getattr(getattr(dataset, "file_meta", None), "TransferSyntaxUID", None)
    if uid is not None and getattr(uid, "is_compressed", False):
        return (
            f"Compressed transfer syntax {uid}; pixel decode failed with {type(exc).__name__}. "
            "Install a compatible codec such as GDCM or pylibjpeg for this environment."
        )
    return f"Pixel decode failed with {type(exc).__name__}: {exc}"


def read_dicom_pixels(path: str | Path) -> DicomReadResult:
    path = str(path)
    try:
        dataset = pydicom.dcmread(path, force=True)
    except Exception as exc:
        raise DicomReadError(f"dcmread failed for {path}: {exc}") from exc

    decode_errors = []
    pixel_array = None
    decoding_plugin = "default"
    for plugin in ("pylibjpeg", ""):
        try:
            if hasattr(dataset, "pixel_array_options"):
                dataset.pixel_array_options(decoding_plugin=plugin)
            pixel_array = dataset.pixel_array
            decoding_plugin = plugin or "default"
            break
        except Exception as exc:
            decode_errors.append(exc)
    if pixel_array is None:
        exc = decode_errors[-1]
        raise DicomReadError(f"{path}: {_codec_hint(dataset, exc)}") from exc

    photometric = getattr(dataset, "PhotometricInterpretation", None)
    normalized = _normalize_pixel_array(pixel_array, dataset)
    normalized = _maybe_invert_monochrome1(normalized, photometric)
    normalized = np.ascontiguousarray(normalized)

    c, t, h, w = normalized.shape
    return DicomReadResult(
        path=path,
        array=normalized,
        original_shape=tuple(np.asarray(pixel_array).shape),
        original_dtype=str(np.asarray(pixel_array).dtype),
        normalized_shape=(c, t, h, w),
        transfer_syntax_uid=get_transfer_syntax_uid(dataset),
        photometric_interpretation=str(photometric) if photometric is not None else None,
        samples_per_pixel=int(getattr(dataset, "SamplesPerPixel", c) or c),
        number_of_frames=int(getattr(dataset, "NumberOfFrames", t) or t),
        channel_count=c,
        frame_count=t,
        height=h,
        width=w,
        decoding_plugin=decoding_plugin,
    )


def sample_video_frames(
    array: np.ndarray,
    num_frames: int | None,
    split: str = "train",
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    if num_frames is None:
        return array

    c, t, h, w = array.shape
    if t == num_frames:
        return array
    if rng is None:
        rng = np.random.default_rng()

    if t > num_frames:
        bins = np.linspace(0, t, num_frames + 1)
        indices: list[int] = []
        for start, end in zip(bins[:-1], bins[1:]):
            left = int(np.floor(start))
            right = max(int(np.ceil(end)) - 1, left)
            if split == "train" and right > left:
                indices.append(int(rng.integers(left, right + 1)))
            else:
                indices.append((left + right) // 2)
        return array[:, indices, :, :]

    pad_count = num_frames - t
    pad_indices = np.full((pad_count,), t - 1, dtype=np.int64)
    return array[:, np.concatenate([np.arange(t), pad_indices]), :, :]


def resize_video_frames(array: np.ndarray, image_size: int | tuple[int, int] | None) -> np.ndarray:
    if image_size is None:
        return array

    if isinstance(image_size, int):
        target_h = target_w = image_size
    else:
        target_h, target_w = image_size

    if array.shape[-2:] == (target_h, target_w):
        return array

    if array.dtype == np.uint8 and array.shape[0] in (1, 3, 4):
        from PIL import Image

        frames = []
        for frame_idx in range(array.shape[1]):
            frame = np.transpose(array[:, frame_idx], (1, 2, 0))
            if frame.shape[-1] == 1:
                frame = frame[..., 0]
            resized = Image.fromarray(frame).resize(
                (target_w, target_h),
                resample=Image.Resampling.BILINEAR,
            )
            resized_array = np.asarray(resized)
            if resized_array.ndim == 2:
                resized_array = resized_array[..., None]
            frames.append(np.transpose(resized_array, (2, 0, 1)))
        return np.ascontiguousarray(np.stack(frames, axis=1))

    import torch
    import torch.nn.functional as F

    tensor = torch.from_numpy(array.astype(np.float32, copy=False)).permute(1, 0, 2, 3)
    tensor = F.interpolate(tensor, size=(target_h, target_w), mode="bilinear", align_corners=False)
    return tensor.permute(1, 0, 2, 3).cpu().numpy()


def adapt_channel_count(array: np.ndarray, target_channels: int | None) -> np.ndarray:
    if target_channels is None:
        return array
    c, t, h, w = array.shape
    if c == target_channels:
        return array
    if target_channels == 1:
        return array.mean(axis=0, keepdims=True)
    if c == 1 and target_channels > 1:
        return np.repeat(array, repeats=target_channels, axis=0)
    if c > target_channels:
        return array[:target_channels]
    repeats = target_channels - c
    tail = np.repeat(array[-1:], repeats=repeats, axis=0)
    return np.concatenate([array, tail], axis=0)


def inspect_dicom_file(path: str | Path) -> dict[str, Any]:
    stats: dict[str, Any] = {
        "path": str(path),
        "dcmread_ok": 0,
        "pixel_array_ok": 0,
        "transfer_syntax_uid": None,
        "photometric_interpretation": None,
        "channel_count": None,
        "frame_count": None,
        "height": None,
        "width": None,
        "original_shape": None,
        "original_dtype": None,
        "normalized_shape": None,
        "decoding_plugin": None,
        "min": None,
        "max": None,
        "mean": None,
        "std": None,
        "error": "",
    }

    try:
        dataset = pydicom.dcmread(str(path), force=True)
        stats["dcmread_ok"] = 1
        stats["transfer_syntax_uid"] = get_transfer_syntax_uid(dataset)
        stats["photometric_interpretation"] = getattr(dataset, "PhotometricInterpretation", None)
        try:
            result = read_dicom_pixels(path)
        except Exception as exc:
            stats["error"] = str(exc)
            return stats

        array = result.array.astype(np.float64, copy=False)
        stats["pixel_array_ok"] = 1
        stats["channel_count"] = result.channel_count
        stats["frame_count"] = result.frame_count
        stats["height"] = result.height
        stats["width"] = result.width
        stats["original_shape"] = list(result.original_shape)
        stats["original_dtype"] = result.original_dtype
        stats["normalized_shape"] = list(result.normalized_shape)
        stats["decoding_plugin"] = result.decoding_plugin
        stats["min"] = float(array.min())
        stats["max"] = float(array.max())
        stats["mean"] = float(array.mean())
        stats["std"] = float(array.std())
        return stats
    except Exception as exc:
        stats["error"] = str(exc)
        return stats
