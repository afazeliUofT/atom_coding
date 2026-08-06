from __future__ import annotations

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


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


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
    values_list = [int(v) for v in values]
    if not values_list:
        return np.empty((0, length), dtype=np.uint8)
    return np.vstack([np.fromiter(((v >> i) & 1 for i in range(length)), dtype=np.uint8, count=length) for v in values_list])


def masks_of_weight(length: int, weight: int):
    import itertools

    if weight < 0 or weight > length:
        return
    for positions in itertools.combinations(range(length), weight):
        mask = 0
        for position in positions:
            mask |= 1 << position
        yield mask


def insert_hidden_bits(base: int, base_length: int, positions: Sequence[int], hidden: int) -> int:
    position_set = set(int(v) for v in positions)
    result = 0
    base_index = 0
    hidden_index = 0
    total_length = base_length + len(positions)
    for output_position in range(total_length):
        if output_position in position_set:
            bit = (int(hidden) >> hidden_index) & 1
            hidden_index += 1
        else:
            bit = (int(base) >> base_index) & 1
            base_index += 1
        result |= bit << output_position
    return result


def delete_positions(word: int, positions: Sequence[int], length: int) -> int:
    positions_set = set(int(v) for v in positions)
    out = 0
    out_index = 0
    for i in range(length):
        if i in positions_set:
            continue
        out |= ((int(word) >> i) & 1) << out_index
        out_index += 1
    return out


def h2(p: float) -> float:
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -p * math.log2(p) - (1.0 - p) * math.log2(1.0 - p)


def percentile(values: Sequence[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=float), q)) if len(values) else float("nan")


def safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator) / max(float(denominator), 1e-15)


def slope_log2(xs: Sequence[float], ys: Sequence[float]) -> float:
    x = np.asarray(xs, dtype=float)
    y = np.maximum(np.asarray(ys, dtype=float), 1e-15)
    if len(x) < 2:
        return float("nan")
    return float(np.polyfit(x, np.log2(y), 1)[0])


def environment_payload() -> dict[str, Any]:
    packages: dict[str, str] = {}
    for name in ("numpy", "scipy", "pandas", "matplotlib", "psutil"):
        try:
            module = __import__(name)
            packages[name] = str(module.__version__)
        except Exception as exc:  # pragma: no cover
            packages[name] = f"unavailable: {exc}"
    return {
        "utc": dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "packages": packages,
    }


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def result_hash_manifest(results_dir: Path, destination: Path) -> None:
    lines: list[str] = []
    for path in sorted(results_dir.iterdir()):
        if path.is_file() and path.name != destination.name:
            lines.append(f"{sha256_file(path)}  {path.name}")
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")
