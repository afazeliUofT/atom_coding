from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import os
import platform
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from scipy.special import gammaln, logsumexp

LN2 = math.log(2.0)


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
    return np.vstack([
        np.fromiter(((v >> i) & 1 for i in range(length)), dtype=np.uint8, count=length)
        for v in values_list
    ])


def masks_of_weight(length: int, weight: int):
    import itertools

    if weight < 0 or weight > length:
        return
    for positions in itertools.combinations(range(length), weight):
        mask = 0
        for position in positions:
            mask |= 1 << position
        yield mask


def insert_bit(base: int, position: int, bit: int, base_length: int) -> int:
    if not (0 <= position <= base_length):
        raise ValueError("invalid insertion position")
    lower = int(base) & ((1 << position) - 1)
    upper = int(base) >> position
    return lower | ((int(bit) & 1) << position) | (upper << (position + 1))


def insert_hidden_bits(base: int, base_length: int, positions: Sequence[int], hidden: int) -> int:
    positions_tuple = tuple(sorted(int(v) for v in positions))
    result = 0
    base_index = 0
    hidden_index = 0
    pos_index = 0
    total_length = base_length + len(positions_tuple)
    for output_position in range(total_length):
        if pos_index < len(positions_tuple) and output_position == positions_tuple[pos_index]:
            bit = (int(hidden) >> hidden_index) & 1
            hidden_index += 1
            pos_index += 1
        else:
            bit = (int(base) >> base_index) & 1
            base_index += 1
        result |= bit << output_position
    return result


def delete_bit(word: int, position: int, length: int) -> int:
    if not (0 <= position < length):
        raise ValueError("invalid deletion position")
    lower = int(word) & ((1 << position) - 1)
    upper = int(word) >> (position + 1)
    return lower | (upper << position)


def delete_positions(word: int, positions: Sequence[int], length: int) -> int:
    result = int(word)
    current_length = int(length)
    for position in sorted((int(v) for v in positions), reverse=True):
        result = delete_bit(result, position, current_length)
        current_length -= 1
    return result


def h2(p: float) -> float:
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -p * math.log2(p) - (1.0 - p) * math.log2(1.0 - p)


def binary_renyi(p: float, alpha: float) -> float:
    if p <= 0.0 or p >= 1.0:
        return 0.0
    if abs(alpha - 1.0) <= 1e-14:
        return h2(p)
    return math.log2(p**alpha + (1.0 - p) ** alpha) / (1.0 - alpha)


def binary_kl(q: float, p: float) -> float:
    if q < 0.0 or q > 1.0 or p <= 0.0 or p >= 1.0:
        if q == p:
            return 0.0
        return float("inf")
    total = 0.0
    if q > 0.0:
        total += q * math.log2(q / p)
    if q < 1.0:
        total += (1.0 - q) * math.log2((1.0 - q) / (1.0 - p))
    return total


def log_binom_coeffs(n: int) -> np.ndarray:
    k = np.arange(n + 1, dtype=float)
    return gammaln(n + 1.0) - gammaln(k + 1.0) - gammaln(n - k + 1.0)


def log_cumulative_binom_counts(n: int) -> np.ndarray:
    logs = log_binom_coeffs(n)
    out = np.empty(n + 1, dtype=float)
    running = -np.inf
    for i, value in enumerate(logs):
        running = np.logaddexp(running, value)
        out[i] = running
    return out


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
    for name in ("numpy", "scipy", "pandas", "matplotlib", "psutil", "pytest"):
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
