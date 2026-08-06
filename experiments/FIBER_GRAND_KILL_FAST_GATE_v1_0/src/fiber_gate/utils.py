from __future__ import annotations

import dataclasses
import datetime as dt
import hashlib
import json
import math
import os
import platform
import subprocess
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def binary_tuple(value: int, length: int) -> tuple[int, ...]:
    return tuple((int(value) >> i) & 1 for i in range(length))


def tuple_to_int(bits: Sequence[int]) -> int:
    value = 0
    for i, bit in enumerate(bits):
        value |= (int(bit) & 1) << i
    return value


def int_array(values: Iterable[int], length: int) -> np.ndarray:
    values = list(int(v) for v in values)
    if not values:
        return np.empty((0, length), dtype=np.uint8)
    shifts = np.arange(length, dtype=np.uint64)
    raw = np.asarray(values, dtype=np.uint64)[:, None]
    return ((raw >> shifts) & 1).astype(np.uint8)


def percentile(values: Sequence[float], q: float) -> float:
    if len(values) == 0:
        return float("nan")
    return float(np.percentile(np.asarray(values, dtype=float), q))


def safe_mean(values: Sequence[float]) -> float:
    if len(values) == 0:
        return float("nan")
    return float(np.mean(np.asarray(values, dtype=float)))


def safe_median(values: Sequence[float]) -> float:
    if len(values) == 0:
        return float("nan")
    return float(np.median(np.asarray(values, dtype=float)))


def fit_log2_slope(xs: Sequence[float], ys: Sequence[float]) -> float:
    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    if len(x) < 2 or np.any(y <= 0):
        return float("nan")
    design = np.column_stack([x, np.ones_like(x)])
    slope, _ = np.linalg.lstsq(design, np.log2(y), rcond=None)[0]
    return float(slope)


def bootstrap_log2_slope(
    xs: Sequence[float],
    grouped_values: Sequence[Sequence[float]],
    rng: np.random.Generator,
    replicates: int = 200,
) -> tuple[float, float, float]:
    means = [safe_mean(group) for group in grouped_values]
    point = fit_log2_slope(xs, means)
    if len(xs) < 2 or replicates <= 0:
        return point, point, point
    samples: list[float] = []
    for _ in range(replicates):
        boot_means = []
        for group in grouped_values:
            arr = np.asarray(group, dtype=float)
            if len(arr) == 0:
                boot_means.append(float("nan"))
            else:
                indices = rng.integers(0, len(arr), size=len(arr))
                boot_means.append(float(np.mean(arr[indices])))
        if np.all(np.isfinite(boot_means)) and np.all(np.asarray(boot_means) > 0):
            samples.append(fit_log2_slope(xs, boot_means))
    if not samples:
        return point, point, point
    return point, percentile(samples, 2.5), percentile(samples, 97.5)


def environment_snapshot() -> dict[str, Any]:
    packages: dict[str, str] = {}
    for name in ("numpy", "scipy", "pandas", "matplotlib", "psutil"):
        try:
            module = __import__(name)
            packages[name] = str(getattr(module, "__version__", "unknown"))
        except Exception as exc:  # pragma: no cover - diagnostic only
            packages[name] = f"unavailable: {exc}"
    def command_version(command: Sequence[str]) -> str | None:
        try:
            result = subprocess.run(
                list(command), text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False
            )
        except FileNotFoundError:
            return None
        text = result.stdout.strip().splitlines()
        return text[0] if text else None
    memory_total = None
    try:
        import psutil

        memory_total = int(psutil.virtual_memory().total)
    except Exception:
        pass
    return {
        "utc": utc_now(),
        "python": platform.python_version(),
        "python_full": platform.python_build(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "memory_total_bytes": memory_total,
        "git": command_version(["git", "--version"]),
        "gcc": command_version(["gcc", "--version"]),
        "packages": packages,
    }


def dataclass_dict(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return {key: dataclass_dict(item) for key, item in dataclasses.asdict(value).items()}
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, dict):
        return {str(key): dataclass_dict(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [dataclass_dict(item) for item in value]
    return value


def logsumexp2(values: Sequence[float]) -> float:
    if not values:
        return float("-inf")
    arr = np.asarray(values, dtype=float)
    maximum = float(np.max(arr))
    if not np.isfinite(maximum):
        return maximum
    return maximum + math.log2(float(np.sum(np.exp2(arr - maximum))))
