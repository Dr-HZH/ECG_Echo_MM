from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


class WfdbReadError(RuntimeError):
    """Raised when a WFDB record cannot be loaded."""


@dataclass
class WfdbReadResult:
    path: str
    signal: np.ndarray
    fs: float
    sig_len: int
    n_sig: int
    sig_name: list[str]
    units: list[str]
    nan_count: int
    inf_count: int
    zero_leads: list[str]
    missing_leads: list[str]


def _import_wfdb():
    try:
        import wfdb  # type: ignore
    except Exception as exc:
        raise WfdbReadError(
            "wfdb is required for ECG record loading. Install wfdb in the active environment."
        ) from exc
    return wfdb


def _normalize_signal_shape(signal: np.ndarray, sig_len: int, n_sig: int) -> np.ndarray:
    arr = np.asarray(signal)
    if arr.ndim != 2:
        raise WfdbReadError(f"Unsupported ECG signal rank: {arr.ndim}")
    if arr.shape == (sig_len, n_sig):
        return arr.T
    if arr.shape == (n_sig, sig_len):
        return arr
    if arr.shape[0] == sig_len:
        return arr.T
    if arr.shape[1] == sig_len:
        return arr
    raise WfdbReadError(f"Unexpected ECG signal shape {arr.shape} for sig_len={sig_len}, n_sig={n_sig}")


def reorder_leads(
    signal: np.ndarray,
    sig_name: list[str],
    lead_order: list[str] | None,
) -> tuple[np.ndarray, list[str]]:
    if not lead_order:
        return signal, []

    name_to_idx = {name: idx for idx, name in enumerate(sig_name)}
    reordered = []
    missing = []
    for lead in lead_order:
        if lead in name_to_idx:
            reordered.append(signal[name_to_idx[lead]])
        else:
            reordered.append(np.zeros(signal.shape[1], dtype=signal.dtype))
            missing.append(lead)
    return np.stack(reordered, axis=0), missing


def resample_signal(signal: np.ndarray, orig_fs: float, target_fs: float) -> np.ndarray:
    if orig_fs == target_fs:
        return signal
    if orig_fs <= 0 or target_fs <= 0:
        raise WfdbReadError(f"Invalid sampling rate conversion {orig_fs} -> {target_fs}")

    old_length = signal.shape[1]
    new_length = int(round(old_length * float(target_fs) / float(orig_fs)))
    x_old = np.linspace(0.0, 1.0, old_length)
    x_new = np.linspace(0.0, 1.0, new_length)
    out = np.empty((signal.shape[0], new_length), dtype=np.float32)
    for idx in range(signal.shape[0]):
        out[idx] = np.interp(x_new, x_old, signal[idx].astype(np.float64, copy=False))
    return out


def crop_or_pad_signal(
    signal: np.ndarray,
    target_length: int | None,
    split: str = "train",
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    if target_length is None:
        return signal

    _, length = signal.shape
    if length == target_length:
        return signal
    if rng is None:
        rng = np.random.default_rng()

    if length > target_length:
        max_start = length - target_length
        if split == "train" and max_start > 0:
            start = int(rng.integers(0, max_start + 1))
        else:
            start = max_start // 2
        return signal[:, start : start + target_length]

    pad = np.zeros((signal.shape[0], target_length - length), dtype=signal.dtype)
    return np.concatenate([signal, pad], axis=1)


def read_wfdb_record(
    record_path: str | Path,
    target_fs: float | None = None,
    target_length: int | None = None,
    split: str = "train",
    lead_order: list[str] | None = None,
    resample_if_needed: bool = False,
    rng: np.random.Generator | None = None,
) -> WfdbReadResult:
    wfdb = _import_wfdb()
    path = str(record_path)
    try:
        record = wfdb.rdrecord(path)
    except Exception as exc:
        raise WfdbReadError(f"rdrecord failed for {path}: {exc}") from exc

    signal = getattr(record, "p_signal", None)
    if signal is None:
        raise WfdbReadError(f"No p_signal available for {path}")

    fs = float(getattr(record, "fs", 0.0) or 0.0)
    sig_len = int(getattr(record, "sig_len", 0) or 0)
    n_sig = int(getattr(record, "n_sig", 0) or 0)
    sig_name = [str(x) for x in list(getattr(record, "sig_name", []) or [])]
    units = [str(x) for x in list(getattr(record, "units", []) or [])]

    signal = _normalize_signal_shape(np.asarray(signal), sig_len=sig_len, n_sig=n_sig)
    signal = signal.astype(np.float32, copy=False)
    signal, missing_leads = reorder_leads(signal, sig_name, lead_order)

    if target_fs is not None and fs > 0 and fs != target_fs:
        if not resample_if_needed:
            raise WfdbReadError(
                f"Sampling rate mismatch for {path}: fs={fs}, target_fs={target_fs}, "
                "set resample_if_needed=True to enable resampling."
            )
        signal = resample_signal(signal, orig_fs=fs, target_fs=float(target_fs))
        fs = float(target_fs)
        sig_len = int(signal.shape[1])

    signal = crop_or_pad_signal(signal, target_length=target_length, split=split, rng=rng)
    nan_count = int(np.isnan(signal).sum())
    inf_count = int(np.isinf(signal).sum())
    zero_mask = np.all(np.isclose(signal, 0.0), axis=1)
    zero_leads = [
        sig_name[idx] if idx < len(sig_name) else f"lead_{idx}"
        for idx, is_zero in enumerate(zero_mask)
        if is_zero
    ]

    return WfdbReadResult(
        path=path,
        signal=signal,
        fs=fs,
        sig_len=int(signal.shape[1]),
        n_sig=int(signal.shape[0]),
        sig_name=sig_name,
        units=units,
        nan_count=nan_count,
        inf_count=inf_count,
        zero_leads=zero_leads,
        missing_leads=missing_leads,
    )


def inspect_wfdb_record(path: str | Path) -> dict[str, Any]:
    stats: dict[str, Any] = {
        "path": str(path),
        "rdrecord_ok": 0,
        "fs": None,
        "sig_len": None,
        "n_sig": None,
        "sig_name": None,
        "units": None,
        "nan_count": None,
        "inf_count": None,
        "zero_lead_count": None,
        "zero_leads": None,
        "q01": None,
        "q05": None,
        "q50": None,
        "q95": None,
        "q99": None,
        "min": None,
        "max": None,
        "error": "",
    }

    try:
        result = read_wfdb_record(path)
        signal = result.signal.astype(np.float64, copy=False)
        stats["rdrecord_ok"] = 1
        stats["fs"] = result.fs
        stats["sig_len"] = result.sig_len
        stats["n_sig"] = result.n_sig
        stats["sig_name"] = "|".join(result.sig_name)
        stats["units"] = "|".join(result.units)
        stats["nan_count"] = result.nan_count
        stats["inf_count"] = result.inf_count
        stats["zero_lead_count"] = len(result.zero_leads)
        stats["zero_leads"] = "|".join(result.zero_leads)
        stats["q01"] = float(np.quantile(signal, 0.01))
        stats["q05"] = float(np.quantile(signal, 0.05))
        stats["q50"] = float(np.quantile(signal, 0.50))
        stats["q95"] = float(np.quantile(signal, 0.95))
        stats["q99"] = float(np.quantile(signal, 0.99))
        stats["min"] = float(signal.min())
        stats["max"] = float(signal.max())
        return stats
    except Exception as exc:
        stats["error"] = str(exc)
        return stats
