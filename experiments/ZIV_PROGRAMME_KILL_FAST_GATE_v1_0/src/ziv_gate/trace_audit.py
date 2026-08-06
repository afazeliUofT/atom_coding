from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from .codelengths import ctw_codelength_bits, entropy_rate_markov, lz78_codelength_bits, renyi_half_rate_markov
from .processes import DEFAULT_REGIMES, fit_markov
from .utils import write_json


def h2(p: float) -> float:
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -p * math.log2(p) - (1.0-p) * math.log2(1.0-p)


def _load_trace(path: Path) -> tuple[int, ...]:
    if path.suffix.lower() == ".npy":
        array = np.load(path)
        bits = tuple(int(v) for v in np.asarray(array).reshape(-1))
    else:
        text = path.read_text(encoding="utf-8", errors="ignore")
        bits = tuple(int(ch) for ch in text if ch in "01")
    if not bits or any(bit not in (0, 1) for bit in bits):
        raise ValueError(f"No valid binary trace in {path}")
    return bits


def synthetic_entropy_audit(rates: Sequence[float]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for regime in DEFAULT_REGIMES:
        pi1 = regime.stationary_one
        marginal = h2(pi1)
        entropy_rate = entropy_rate_markov(regime.p01, regime.p10)
        renyi = renyi_half_rate_markov(regime.p01, regime.p10)
        row: dict[str, Any] = {
            "source": regime.name,
            "structured": regime.structured,
            "p01": regime.p01,
            "p10": regime.p10,
            "stationary_one": pi1,
            "marginal_entropy": marginal,
            "shannon_entropy_rate": entropy_rate,
            "renyi_half_rate": renyi,
            "renyi_to_marginal_ratio": renyi / max(marginal, 1e-12),
            "proposal_ratio_gate": renyi <= 0.6*marginal,
        }
        for rate in rates:
            row[f"query_feasible_R{rate}"] = renyi <= 1.0-rate
            row[f"capacity_feasible_R{rate}"] = entropy_rate < 1.0-rate
        rows.append(row)
    return rows


def real_trace_audit(trace_dir: Path, rates: Sequence[float], ctw_depth: int, max_symbols: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(p for p in trace_dir.glob("*") if p.is_file() and not p.name.startswith(".")):
        try:
            bits = _load_trace(path)[:max_symbols]
        except Exception as exc:
            rows.append({"file": path.name, "error": str(exc)})
            continue
        pi1 = sum(bits)/len(bits)
        p01, p10 = fit_markov(bits)
        marginal = h2(pi1)
        entropy_rate = entropy_rate_markov(p01, p10)
        renyi = renyi_half_rate_markov(p01, p10)
        row: dict[str, Any] = {
            "file": path.name,
            "symbols": len(bits),
            "empirical_one_fraction": pi1,
            "fitted_p01": p01,
            "fitted_p10": p10,
            "marginal_entropy": marginal,
            "fitted_markov_entropy_rate": entropy_rate,
            "fitted_markov_renyi_half_rate": renyi,
            "lz78_bits_per_symbol": lz78_codelength_bits(bits)/len(bits),
            "ctw_bits_per_symbol": ctw_codelength_bits(bits,depth=ctw_depth)/len(bits),
            "proposal_ratio_gate": renyi <= 0.6*marginal,
        }
        for rate in rates:
            row[f"query_feasible_R{rate}"] = renyi <= 1.0-rate
            row[f"capacity_feasible_R{rate}"] = entropy_rate < 1.0-rate
        rows.append(row)
    return rows


def run_trace_gate(output_dir: Path, package_root: Path, rates: Sequence[float], ctw_depth: int, max_symbols: int) -> dict[str, Any]:
    synthetic = synthetic_entropy_audit(rates)
    real = real_trace_audit(package_root/"data"/"real_traces", rates, ctw_depth, max_symbols)
    pd.DataFrame(synthetic).to_csv(output_dir/"07_synthetic_entropy_audit.csv",index=False)
    if real:
        pd.DataFrame(real).to_csv(output_dir/"07_real_trace_audit.csv",index=False)
    valid_real = [row for row in real if "error" not in row]
    real_pass_count = sum(bool(row.get("proposal_ratio_gate")) for row in valid_real)
    payload = {
        "synthetic": synthetic,
        "real_trace_files_found": len(real),
        "valid_real_traces": len(valid_real),
        "real_traces_passing_ratio_gate": real_pass_count,
        "manual_systems_gate": "PASS" if len(valid_real)>=2 and real_pass_count>=2 else "PENDING_REAL_POST_FRONT_END_TRACES",
        "pass": bool(len(valid_real)>=2 and real_pass_count>=2),
        "warning": (
            "LZ/CTW log loss estimates Shannon-type compressibility; they are not direct estimators of the Renyi-1/2 rate. "
            "The gate computes the latter from a fitted order-1 Markov model and labels that modeling assumption explicitly."
        ),
    }
    write_json(output_dir/"07_trace_gate.json",payload)
    return payload
