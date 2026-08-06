from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import pandas as pd
from scipy.optimize import brentq

from .utils import binary_renyi, h2, write_json


def threshold_for_target(target: float, metric: str) -> float:
    if not (0.0 < target < 1.0):
        raise ValueError("target must be in (0,1)")
    if metric == "typical":
        fn = h2
    elif metric == "mean":
        fn = lambda p: binary_renyi(p, 0.5)
    else:
        raise ValueError(metric)
    return float(brentq(lambda p: fn(p) - target, 1e-15, 0.5 - 1e-15))


def run_phase_diagram_gate(output_dir: Path, rates: Sequence[float], p_grid: Sequence[float]) -> dict[str, Any]:
    rows = []
    operating = []
    for rate in rates:
        redundancy = 1.0 - float(rate)
        p_typical = threshold_for_target(redundancy, "typical")
        p_mean = threshold_for_target(redundancy, "mean")
        rows.append(
            {
                "rate": float(rate),
                "redundancy_exponent": redundancy,
                "typical_threshold_p_h2_equals_1_minus_R": p_typical,
                "mean_threshold_p_Hhalf_equals_1_minus_R": p_mean,
                "mean_threshold_below_typical": p_mean < p_typical,
            }
        )
        for p in p_grid:
            operating.append(
                {
                    "rate": float(rate),
                    "p": float(p),
                    "h2_p": h2(float(p)),
                    "Hhalf_p": binary_renyi(float(p), 0.5),
                    "typical_fiber_favorable": h2(float(p)) < redundancy,
                    "mean_fiber_favorable": binary_renyi(float(p), 0.5) < redundancy,
                    "hybrid_typical_exponent": min(h2(float(p)), redundancy),
                    "hybrid_mean_exponent": min(binary_renyi(float(p), 0.5), redundancy),
                }
            )
    frame = pd.DataFrame(rows)
    op = pd.DataFrame(operating)
    frame.to_csv(output_dir / "03_phase_thresholds.csv", index=False)
    op.to_csv(output_dir / "03_operating_regions.csv", index=False)
    payload = {
        "thresholds": rows,
        "mean_threshold_below_typical_all": bool(frame["mean_threshold_below_typical"].all()),
        "interpretation": (
            "A fixed-quantile FIBER/search crossover is predicted by h2(p)=1-R, while the mean-work crossover "
            "is predicted by H_{1/2}(p)=1-R. The gap explains why a decoder can win in median latency while "
            "losing in average or extreme-tail work."
        ),
        "hybrid_theorem": (
            "If an independent exact code-side search uses poly(n) 2^{n(1-R)} work, a parallel hybrid obeys "
            "typical exponent at most min(h2(p),1-R) and rho-moment exponent at most "
            "min(rho H_{1/(1+rho)}(p), rho(1-R))."
        ),
    }
    write_json(output_dir / "03_phase_diagram_gate.json", payload)
    return payload
