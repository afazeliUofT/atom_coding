from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import random
import subprocess
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def seed_everything(seed: int) -> np.random.Generator:
    random.seed(seed)
    np.random.seed(seed)
    return np.random.default_rng(seed)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def json_default(obj: Any) -> Any:
    if is_dataclass(obj):
        return asdict(obj)
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, set):
        return sorted(obj)
    raise TypeError(f"Cannot serialize {type(obj)}")


def write_json(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=json_default) + "\n",
        encoding="utf-8",
    )


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def build_sha256_manifest(root: Path, output: Path, exclude: Iterable[str] = ()) -> None:
    excluded = set(exclude)
    lines: list[str] = []
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        rel = path.relative_to(root).as_posix()
        if rel in excluded or rel == output.relative_to(root).as_posix():
            continue
        if any(part in {"__pycache__", ".pytest_cache"} for part in path.parts):
            continue
        lines.append(f"{sha256_file(path)}  {rel}")
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def environment_record() -> dict[str, Any]:
    record: dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
    }
    try:
        import numpy
        import scipy
        import pandas
        import matplotlib
        import psutil

        record["packages"] = {
            "numpy": numpy.__version__,
            "scipy": scipy.__version__,
            "pandas": pandas.__version__,
            "matplotlib": matplotlib.__version__,
            "psutil": psutil.__version__,
        }
        vm = psutil.virtual_memory()
        record["memory_total_bytes"] = int(vm.total)
    except Exception as exc:  # pragma: no cover - diagnostic only
        record["package_probe_error"] = repr(exc)
    for cmd, key in [(["git", "--version"], "git"), (["gcc", "--version"], "gcc")]:
        try:
            cp = subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=10)
            record[key] = (cp.stdout or cp.stderr).splitlines()[0]
        except Exception as exc:  # pragma: no cover
            record[f"{key}_error"] = repr(exc)
    return record


def binary_entropy(p: float) -> float:
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -p * math.log2(p) - (1.0 - p) * math.log2(1.0 - p)


def stable_log2(x: float, floor: float = 1e-300) -> float:
    return math.log2(max(float(x), floor))


def percentile(values: Iterable[float], p: float) -> float:
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return float("nan")
    return float(np.percentile(arr, p))
