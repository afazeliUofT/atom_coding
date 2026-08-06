from __future__ import annotations

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


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_manifest(root: Path, output: Path, exclude: Iterable[str] = ()) -> None:
    excluded = set(exclude)
    rows: list[str] = []
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        rel = path.relative_to(root).as_posix()
        if rel in excluded or rel == output.relative_to(root).as_posix():
            continue
        rows.append(f"{sha256_file(path)}  {rel}")
    output.write_text("\n".join(rows) + "\n", encoding="utf-8")


def logsumexp2(values: Sequence[float] | np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return float("-inf")
    maximum = float(np.max(arr))
    if not math.isfinite(maximum):
        return maximum
    return maximum + math.log2(float(np.sum(np.exp2(arr - maximum))))


def stable_log2_one_minus_power(log2_bad_probability: float, competitors_log2: float) -> float:
    """Return log2(error probability) for 1-(1-p)^m from log2 p and log2 m.

    The result is exact up to floating precision in the ranges used by the gate.
    """
    if log2_bad_probability == float("-inf"):
        return float("-inf")
    if log2_bad_probability >= 0.0:
        return 0.0
    log_mp = competitors_log2 + log2_bad_probability
    if log_mp < -40.0:
        return log_mp / math.log(2.0) * math.log(2.0)  # identity; keeps intent explicit
    p = 2.0 ** log2_bad_probability
    if competitors_log2 > 50.0:
        # Poisson approximation is extremely accurate when p is tiny and m is huge.
        lam = 2.0 ** min(competitors_log2 + math.log2(max(p, 1e-300)), 50.0)
        if lam >= 40.0:
            return 0.0
        value = -math.expm1(-lam)
        return math.log2(max(value, 1e-300))
    m = max(1, int(round(2.0 ** competitors_log2)))
    value = -math.expm1(m * math.log1p(-p))
    return math.log2(max(value, 1e-300))


def random_coding_bler_from_log2_level_rank(log2_rank: float, n: int, rate: float) -> float:
    """Independent-competitor finite-block proxy under a distinct-codeword ensemble.

    Bad competitor fraction is (rank-1)/(2^n-1). The power model uses independent
    competitors sampled uniformly from the remaining ambient vectors. This is the
    with-replacement analogue of the exact hypergeometric law and has the same exponent.
    """
    if log2_rank <= 0.0:
        return 0.0
    # log2(rank - 1), accurately enough for rank represented only through logarithms.
    if log2_rank > 40.0:
        log2_bad_count = log2_rank
    else:
        rank = max(1.0, 2.0 ** log2_rank)
        bad = max(0.0, rank - 1.0)
        if bad == 0.0:
            return 0.0
        log2_bad_count = math.log2(bad)
    log2_denominator = math.log2(max(1.0, 2.0**n - 1.0)) if n <= 1023 else float(n)
    log2_p = min(0.0, log2_bad_count - log2_denominator)
    log2_m = max(0.0, rate * n)  # M-1 is asymptotically 2^(nR)
    if log2_p + log2_m > 35.0:
        return 1.0
    p = 2.0 ** log2_p
    if p >= 1.0 - 1e-15:
        return 1.0
    if log2_m > 50.0:
        lam = 2.0 ** (log2_m + log2_p)
        return float(-math.expm1(-lam))
    m = max(1, int(round(2.0 ** log2_m)) - 1)
    return float(-math.expm1(m * math.log1p(-p)))


def environment_snapshot() -> dict[str, Any]:
    packages: dict[str, str] = {}
    for name in ("numpy", "pandas", "matplotlib", "scipy", "psutil"):
        try:
            module = __import__(name)
            packages[name] = str(module.__version__)
        except Exception:
            packages[name] = "unavailable"
    def version(command: list[str]) -> str:
        try:
            result = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
            return result.stdout.splitlines()[0] if result.stdout else "unavailable"
        except Exception:
            return "unavailable"
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "packages": packages,
        "git": version(["git", "--version"]),
        "gcc": version(["gcc", "--version"]),
    }


def binary_tuple(value: int, n: int) -> tuple[int, ...]:
    return tuple((value >> i) & 1 for i in range(n))


def tuple_to_int(bits: Sequence[int]) -> int:
    value = 0
    for i, bit in enumerate(bits):
        value |= (int(bit) & 1) << i
    return value


def xor_tuples(a: Sequence[int], b: Sequence[int]) -> tuple[int, ...]:
    if len(a) != len(b):
        raise ValueError("Sequences must have equal length")
    return tuple(int(x) ^ int(y) for x, y in zip(a, b, strict=True))


def slope_log2(x: Sequence[float], y: Sequence[float]) -> float:
    xs = np.asarray(x, dtype=float)
    ys = np.asarray(y, dtype=float)
    if xs.size < 2 or np.any(ys <= 0):
        return float("nan")
    return float(np.polyfit(xs, np.log2(ys), 1)[0])


def slope_linear(x: Sequence[float], y: Sequence[float]) -> float:
    xs = np.asarray(x, dtype=float)
    ys = np.asarray(y, dtype=float)
    if xs.size < 2:
        return float("nan")
    return float(np.polyfit(xs, ys, 1)[0])
