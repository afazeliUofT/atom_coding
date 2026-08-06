from __future__ import annotations

import glob
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .utils import binary_tuple, tuple_to_int, write_json


def simulate_timing_slip_bits(
    bits: tuple[int, ...],
    snr_db: float,
    isi_strength: float,
    rng: np.random.Generator,
) -> tuple[int, int, np.ndarray]:
    """Synthetic post-timing-recovery output with one positive cycle slip.

    A uniformly random initial phase and one-symbol accumulated clock drift over
    the frame produce one skipped input decision.  A small phase-dependent
    neighbour leakage and AWGN create hard-decision substitutions.  The output
    is a received word of length n-1, the deleted position, and the binary error
    vector relative to the surviving input symbols.
    """
    n = len(bits)
    if n < 3:
        raise ValueError("n must be at least 3")
    phase0 = float(rng.random())
    symbols = 1.0 - 2.0 * np.asarray(bits, dtype=float)
    sigma = math.sqrt(1.0 / (2.0 * 10.0 ** (snr_db / 10.0)))
    indices: list[int] = []
    observations: list[int] = []
    errors: list[int] = []
    for k in range(n - 1):
        phase = phase0 + k / (n - 1)
        shift = 1 if phase >= 1.0 else 0
        frac = phase - math.floor(phase)
        idx = k + shift
        idx = min(idx, n - 1)
        indices.append(idx)
        neighbour = min(n - 1, idx + 1)
        leak = isi_strength * (2.0 * abs(frac - 0.5))
        sample = (1.0 - leak) * symbols[idx] + leak * symbols[neighbour] + sigma * float(rng.normal())
        bit_hat = int(sample < 0.0)
        observations.append(bit_hat)
        errors.append(int(bit_hat != bits[idx]))
    missing = sorted(set(range(n)).difference(indices))
    if len(missing) != 1:
        raise AssertionError(f"timing model did not create exactly one slip: {missing}")
    return tuple_to_int(observations), int(missing[0]), np.asarray(errors, dtype=np.uint8)


def _lag1_correlation(error_rows: list[np.ndarray]) -> float:
    left: list[float] = []
    right: list[float] = []
    for row in error_rows:
        if len(row) >= 2:
            left.extend(row[:-1].astype(float))
            right.extend(row[1:].astype(float))
    if not left or np.std(left) <= 1e-15 or np.std(right) <= 1e-15:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def _trace_audit(files: list[str]) -> dict[str, Any]:
    if not files:
        return {
            "status": "PENDING_MEASURED_REAL_TRACE",
            "files": [],
            "valid_rows": 0,
            "message": "No measured CSV trace was bundled. The scientific classification must retain REAL_TRACE_REQUIRED.",
        }
    frames = []
    for filename in files:
        frame = pd.read_csv(filename)
        required = {"n", "deleted_position", "substitution_weight"}
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{filename} missing columns {sorted(missing)}")
        frame["trace_file"] = Path(filename).name
        frames.append(frame)
    data = pd.concat(frames, ignore_index=True)
    valid = (
        (data["n"] >= 2)
        & (data["deleted_position"] >= 0)
        & (data["deleted_position"] < data["n"])
        & (data["substitution_weight"] >= 0)
        & (data["substitution_weight"] <= data["n"] - 1)
    )
    return {
        "status": "MEASURED_TRACE_PRESENT",
        "files": [Path(v).name for v in files],
        "valid_rows": int(valid.sum()),
        "total_rows": int(len(data)),
        "all_rows_valid": bool(valid.all()),
        "mean_substitution_rate": float((data["substitution_weight"] / (data["n"] - 1)).mean()),
    }


def run_physical_plausibility_gate(
    output_dir: Path,
    rng: np.random.Generator,
    n: int,
    train_frames: int,
    test_frames: int,
    snr_db: float,
    isi_strength: float,
    real_trace_glob: str,
    package_root: Path,
) -> dict[str, Any]:
    def generate(count: int):
        positions: list[int] = []
        errors: list[np.ndarray] = []
        rows: list[dict[str, Any]] = []
        for frame_id in range(count):
            bits = tuple(int(v) for v in rng.integers(0, 2, size=n))
            received, deleted, err = simulate_timing_slip_bits(bits, snr_db, isi_strength, rng)
            positions.append(deleted)
            errors.append(err)
            rows.append(
                {
                    "frame_id": frame_id,
                    "n": n,
                    "deleted_position": deleted,
                    "substitution_weight": int(err.sum()),
                    "substitution_rate": float(err.mean()),
                    "received": int(received),
                }
            )
        return positions, errors, rows

    train_pos, train_err, train_rows = generate(train_frames)
    test_pos, test_err, test_rows = generate(test_frames)
    q_train = np.bincount(train_pos, minlength=n).astype(float)
    q_train /= q_train.sum()
    q_test = np.bincount(test_pos, minlength=n).astype(float)
    q_test /= q_test.sum()
    p_train = float(np.concatenate(train_err).mean())
    p_test = float(np.concatenate(test_err).mean())
    tv = 0.5 * float(np.abs(q_train - q_test).sum())
    lag = _lag1_correlation(test_err)

    pd.DataFrame(train_rows).to_csv(output_dir / "07_physical_train.csv.gz", index=False, compression="gzip")
    pd.DataFrame(test_rows).to_csv(output_dir / "07_physical_test.csv.gz", index=False, compression="gzip")
    pd.DataFrame(
        {
            "position": np.arange(n),
            "q_train": q_train,
            "q_test": q_test,
            "uniform": np.full(n, 1.0 / n),
        }
    ).to_csv(output_dir / "07_physical_position_calibration.csv", index=False)

    files = sorted(glob.glob(str(package_root / real_trace_glob)))
    measured = _trace_audit(files)
    payload = {
        "model": "synthetic clock-recovery cycle slip with phase-dependent ISI and AWGN hard decisions",
        "n": int(n),
        "train_frames": int(train_frames),
        "test_frames": int(test_frames),
        "snr_db": float(snr_db),
        "isi_strength": float(isi_strength),
        "exactly_one_slip_fraction": 1.0,
        "fitted_substitution_probability": p_train,
        "heldout_substitution_probability": p_test,
        "absolute_p_fit_error": abs(p_train - p_test),
        "heldout_deletion_position_tv_to_fitted": tv,
        "heldout_error_lag1_correlation": lag,
        "measured_trace_audit": measured,
        "interpretation": (
            "This is a physically motivated synthetic front-end plausibility test, not measured deployment evidence. "
            "A positive result cannot remove the REAL_TRACE_REQUIRED qualifier."
        ),
    }
    write_json(output_dir / "07_physical_plausibility_gate.json", payload)
    return payload
