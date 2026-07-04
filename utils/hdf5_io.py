from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def _import_h5py():
    try:
        import h5py  # type: ignore
    except Exception as exc:
        raise RuntimeError("h5py is required for HDF5 input. Install it in the active environment.") from exc
    return h5py


class HDF5ShardReader:
    """Process-local HDF5 handle cache suitable for DataLoader workers."""

    def __init__(self, data_root: str | Path | None = None) -> None:
        self.data_root = Path(data_root) if data_root is not None else Path(".")
        self._handles: dict[str, Any] = {}

    def _resolve(self, path: str | Path) -> Path:
        path = Path(path)
        return path if path.is_absolute() else self.data_root / path

    def read(self, h5_path: str | Path, h5_key: str) -> np.ndarray:
        array, _ = self.read_with_attrs(h5_path, h5_key)
        return array

    def read_with_attrs(self, h5_path: str | Path, h5_key: str) -> tuple[np.ndarray, dict[str, Any]]:
        h5py = _import_h5py()
        resolved = str(self._resolve(h5_path))
        handle = self._handles.get(resolved)
        if handle is None:
            handle = h5py.File(resolved, "r", libver="latest", swmr=True)
            self._handles[resolved] = handle
        if h5_key not in handle:
            raise KeyError(f"HDF5 key not found: {resolved}:{h5_key}")
        dataset = handle[h5_key]
        return np.asarray(dataset), {str(key): value for key, value in dataset.attrs.items()}

    def close(self) -> None:
        for handle in self._handles.values():
            handle.close()
        self._handles.clear()

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_handles"] = {}
        return state

    def __del__(self) -> None:
        self.close()


def decode_wfdb_digital(array: np.ndarray, attrs: dict[str, Any]) -> np.ndarray:
    gain = np.asarray(attrs["adc_gain"], dtype=np.float32).reshape(-1, 1)
    baseline = np.asarray(attrs["baseline"], dtype=np.float32).reshape(-1, 1)
    signal = (array.astype(np.float32) - baseline) / gain
    digital_nan = int(attrs.get("digital_nan", -32768))
    signal[array == digital_nan] = np.nan
    return signal
