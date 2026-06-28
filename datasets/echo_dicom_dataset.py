from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from utils.dicom_io import (
    adapt_channel_count,
    read_dicom_pixels,
    resize_video_frames,
    sample_video_frames,
)
from utils.video_transforms import (
    crop_video,
    jitter_brightness_contrast,
    minmax_scale_video,
    normalize_video,
    resize_video,
    translate_video,
)
from utils.hdf5_io import HDF5ShardReader


def _parse_label(value: Any) -> torch.Tensor:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return torch.tensor(float("nan"), dtype=torch.float32)
    try:
        return torch.tensor(float(value), dtype=torch.float32)
    except Exception:
        return torch.tensor(float("nan"), dtype=torch.float32)


def _normalize_echo_npy(array: np.ndarray) -> np.ndarray:
    arr = np.asarray(array)
    if arr.ndim == 2:
        return arr[None, None, :, :]
    if arr.ndim == 3:
        if arr.shape[0] <= 4:
            return arr[:, None, :, :]
        if arr.shape[-1] <= 4:
            return np.transpose(arr, (2, 0, 1))[..., None].transpose(0, 3, 1, 2)
        return arr[None, :, :, :]
    if arr.ndim == 4:
        if arr.shape[0] <= 4 and arr.shape[1] > 4:
            return arr
        if arr.shape[-1] <= 4 and arr.shape[0] > 4:
            return np.transpose(arr, (3, 0, 1, 2))
    raise ValueError(f"Unsupported echo npy shape: {arr.shape}")


class EchoDicomDataset(Dataset):
    def __init__(
        self,
        csv_path: str | Path,
        dataframe: pd.DataFrame | None = None,
        echo_path_col: str = "echo_dcm_path",
        label_col: str | None = None,
        patient_id_col: str | None = None,
        split: str = "train",
        num_frames: int | None = None,
        image_size: int | tuple[int, int] | None = None,
        target_channels: int | None = None,
        intensity_mode: str = "minmax",
        input_format: str = "dicom",
        data_root: str | Path | None = None,
        h5_path_col: str = "h5_path",
        h5_key_col: str = "h5_key",
        h5_root: str | Path | None = None,
        cache_dir: str | Path | None = None,
        cache_image_size: int | tuple[int, int] | None = None,
        crop_scale_range: tuple[float, float] | None = None,
        brightness_jitter: float = 0.0,
        contrast_jitter: float = 0.0,
        translate_fraction: float = 0.0,
        normalize_mean: Sequence[float] | None = None,
        normalize_std: Sequence[float] | None = None,
    ) -> None:
        self.csv_path = Path(csv_path)
        self.df = dataframe.copy().reset_index(drop=True) if dataframe is not None else pd.read_csv(self.csv_path, low_memory=False)
        self.echo_path_col = echo_path_col
        self.label_col = label_col
        self.patient_id_col = patient_id_col
        self.split = split
        self.num_frames = num_frames
        self.image_size = image_size
        self.target_channels = target_channels
        self.intensity_mode = intensity_mode
        self.input_format = input_format
        self.data_root = Path(data_root) if data_root is not None else self.csv_path.parent
        self.h5_path_col = h5_path_col
        self.h5_key_col = h5_key_col
        self.h5_reader = HDF5ShardReader(h5_root) if input_format == "hdf5" else None
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self.cache_image_size = cache_image_size
        self.crop_scale_range = crop_scale_range
        self.brightness_jitter = brightness_jitter
        self.contrast_jitter = contrast_jitter
        self.translate_fraction = translate_fraction
        self.normalize_mean = list(normalize_mean) if normalize_mean is not None else None
        self.normalize_std = list(normalize_std) if normalize_std is not None else None
        self.rng = np.random.default_rng()
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def __len__(self) -> int:
        return len(self.df)

    def _resolve_path(self, value: str) -> Path:
        path = Path(value)
        if path.is_absolute():
            return path
        return self.data_root / value

    def _cache_file_path(self, abs_path: Path) -> Path:
        if self.cache_dir is None:
            raise RuntimeError("Cache directory is not configured.")

        cache_image_size = self.cache_image_size if self.cache_image_size is not None else self.image_size
        if isinstance(cache_image_size, int):
            cache_suffix = f"{cache_image_size}x{cache_image_size}"
        elif cache_image_size is None:
            cache_suffix = "native"
        else:
            cache_suffix = f"{cache_image_size[0]}x{cache_image_size[1]}"

        try:
            relative = abs_path.relative_to(self.data_root)
            parent = relative.parent
            stem = relative.stem
        except ValueError:
            parent = Path()
            stem = abs_path.stem

        channel_suffix = self.target_channels if self.target_channels is not None else "nativec"
        cache_name = f"{stem}_cthw_{channel_suffix}_{cache_suffix}.npy"
        cache_path = self.cache_dir / parent / cache_name
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        return cache_path

    def _load_dicom_with_cache(self, path: Path) -> tuple[np.ndarray, dict[str, Any]]:
        cache_path = self._cache_file_path(path)
        if cache_path.exists():
            cached = np.load(cache_path)
            return cached, {
                "original_dtype": str(cached.dtype),
                "original_shape": list(cached.shape),
                "normalized_shape": list(cached.shape),
                "transfer_syntax_uid": None,
                "photometric_interpretation": None,
                "cache_path": str(cache_path),
                "loaded_from_cache": True,
            }

        result = read_dicom_pixels(path)
        array = result.array
        array = adapt_channel_count(array, target_channels=self.target_channels)
        if self.cache_image_size is not None:
            array = resize_video_frames(array, image_size=self.cache_image_size)

        if np.issubdtype(array.dtype, np.floating):
            array = np.clip(np.rint(array), 0, 255).astype(np.uint8, copy=False)
        else:
            array = np.ascontiguousarray(array)
        np.save(cache_path, array)
        return array, {
            "original_dtype": result.original_dtype,
            "original_shape": list(result.original_shape),
            "normalized_shape": list(array.shape),
            "transfer_syntax_uid": result.transfer_syntax_uid,
            "photometric_interpretation": result.photometric_interpretation,
            "cache_path": str(cache_path),
            "loaded_from_cache": False,
        }

    def _load_array(self, path: Path, h5_key: str | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        if self.input_format == "dicom":
            if self.cache_dir is not None:
                array, meta = self._load_dicom_with_cache(path)
            else:
                result = read_dicom_pixels(path)
                meta = {
                    "original_dtype": result.original_dtype,
                    "original_shape": list(result.original_shape),
                    "normalized_shape": list(result.normalized_shape),
                    "transfer_syntax_uid": result.transfer_syntax_uid,
                    "photometric_interpretation": result.photometric_interpretation,
                    "cache_path": None,
                    "loaded_from_cache": False,
                }
                array = result.array
        elif self.input_format == "npy":
            array = _normalize_echo_npy(np.load(path))
            meta = {
                "original_dtype": str(array.dtype),
                "original_shape": list(array.shape),
                "normalized_shape": list(array.shape),
                "transfer_syntax_uid": None,
                "photometric_interpretation": None,
                "cache_path": None,
                "loaded_from_cache": False,
            }
        elif self.input_format == "hdf5":
            if self.h5_reader is None or h5_key is None:
                raise ValueError("HDF5 input requires h5_path and h5_key.")
            stored = self.h5_reader.read(path, h5_key)
            if stored.ndim != 4 or stored.shape[-1] not in (1, 3, 4):
                raise ValueError(f"Expected HDF5 Echo [T,H,W,C], got {stored.shape}")
            array = np.ascontiguousarray(np.transpose(stored, (3, 0, 1, 2)))
            meta = {
                "original_dtype": str(stored.dtype),
                "original_shape": list(stored.shape),
                "normalized_shape": list(array.shape),
                "transfer_syntax_uid": None,
                "photometric_interpretation": None,
                "cache_path": str(path),
                "loaded_from_cache": True,
                "h5_key": h5_key,
            }
        else:
            raise ValueError(f"Unsupported echo input_format={self.input_format}")

        array = sample_video_frames(array, num_frames=self.num_frames, split=self.split, rng=self.rng)
        if self.cache_dir is None:
            array = adapt_channel_count(array, target_channels=self.target_channels)
        return array, meta

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.df.iloc[index]
        path_col = self.h5_path_col if self.input_format == "hdf5" else self.echo_path_col
        path_value = row[path_col]
        if pd.isna(path_value):
            raise FileNotFoundError(f"Missing echo path at row {index}")

        abs_path = Path(str(path_value)) if self.input_format == "hdf5" else self._resolve_path(str(path_value))
        h5_key = str(row[self.h5_key_col]) if self.input_format == "hdf5" else None
        array, meta = self._load_array(abs_path, h5_key=h5_key)
        tensor = torch.from_numpy(array.astype(np.float32, copy=False))
        if self.intensity_mode == "minmax":
            tensor = minmax_scale_video(tensor)
        elif self.intensity_mode == "uint8_scale":
            tensor = tensor / 255.0
        tensor = crop_video(tensor, split=self.split, crop_scale_range=self.crop_scale_range, rng=self.rng)
        tensor = translate_video(tensor, split=self.split, max_translate_fraction=self.translate_fraction, rng=self.rng)
        tensor = resize_video(tensor, image_size=self.image_size)
        tensor = jitter_brightness_contrast(
            tensor,
            split=self.split,
            brightness=self.brightness_jitter,
            contrast=self.contrast_jitter,
            rng=self.rng,
        )
        tensor = normalize_video(tensor, mean=self.normalize_mean, std=self.normalize_std)

        sample: dict[str, Any] = {
            "echo": tensor,
            "echo_path": str(abs_path),
            "row_index": int(index),
            "meta": meta,
        }
        if self.label_col is not None and self.label_col in row.index:
            sample["label"] = _parse_label(row[self.label_col])
        if self.patient_id_col is not None and self.patient_id_col in row.index:
            sample["patient_id"] = row[self.patient_id_col]
        return sample
