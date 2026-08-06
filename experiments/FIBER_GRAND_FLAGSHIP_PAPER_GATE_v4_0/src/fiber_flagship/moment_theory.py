from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import logsumexp
from scipy.stats import binom

from .utils import (
    LN2,
    binary_kl,
    binary_renyi,
    h2,
    log_binom_coeffs,
    log_cumulative_binom_counts,
    write_json,
)


@dataclass(frozen=True)
class MomentRates:
    reveal_lower: float
    reveal_upper: float
    certificate_upper: float
    theory: float


def fixed_edit_factor(n: int, t: int) -> int:
    return (1 << t) * math.comb(n, t)


def extra_shells(n: int, t: int, p: float) -> int:
    if p <= 0.0:
        return 0
    streams = math.comb(n, t)
    return int(math.floor(math.log(streams, (1.0 - p) / p)))


def _log1p_exp(x: float) -> float:
    if x > 50.0:
        return x
    return math.log1p(math.exp(x))


def exact_moment_rates(n: int, t: int, p: float, rho: float) -> MomentRates:
    if not (0.0 < p < 0.5):
        raise ValueError("moment audit requires 0 < p < 1/2")
    m = n - t
    log_coeff = log_binom_coeffs(m)
    e = np.arange(m + 1, dtype=float)
    log_prob = log_coeff + e * math.log(p) + (m - e) * math.log(1.0 - p)
    log_prob -= logsumexp(log_prob)
    log_cum = log_cumulative_binom_counts(m)
    log_k = math.log(fixed_edit_factor(n, t))

    lower = np.zeros(m + 1, dtype=float)
    lower[0] = 0.0  # log(1)
    for idx in range(1, m + 1):
        lower[idx] = _log1p_exp(log_k + log_cum[idx - 1])
    upper = log_k + log_cum
    ell = extra_shells(n, t, p)
    cert_index = np.minimum(np.arange(m + 1) + ell, m)
    cert = log_k + log_cum[cert_index]

    def rate(log_rank: np.ndarray) -> float:
        return float(logsumexp(log_prob + rho * log_rank) / (n * LN2))

    alpha = 1.0 / (1.0 + rho)
    theory = rho * binary_renyi(p, alpha)
    return MomentRates(rate(lower), rate(upper), rate(cert), theory)


def reveal_quantile_rate(n: int, t: int, p: float, quantile: float, certificate: bool = False) -> float:
    m = n - t
    e_q = int(binom.ppf(quantile, m, p))
    log_cum = log_cumulative_binom_counts(m)
    index = e_q + (extra_shells(n, t, p) if certificate else 0)
    index = min(m, max(0, index))
    return float((math.log(fixed_edit_factor(n, t)) + log_cum[index]) / (n * LN2))


def q_from_binary_entropy(gamma: float) -> float:
    if gamma <= 0.0:
        return 0.0
    if gamma >= 1.0:
        return 0.5
    return float(brentq(lambda q: h2(q) - gamma, 1e-15, 0.5 - 1e-15))


def reveal_ldp_rate(p: float, gamma: float) -> float:
    if gamma < 0.0 or gamma > 1.0:
        return float("inf")
    if gamma == 1.0:
        return binary_kl(0.5, p)
    return binary_kl(q_from_binary_entropy(gamma), p)


def exponential_tail_quantile_exponent(p: float, beta: float) -> float:
    if beta <= 0.0:
        return h2(p)
    maximum = binary_kl(0.5, p)
    if beta >= maximum:
        return 1.0
    q = brentq(lambda x: binary_kl(x, p) - beta, p, 0.5)
    return h2(float(q))


def run_moment_tail_gate(
    output_dir: Path,
    blocklengths: Sequence[int],
    probabilities: Sequence[float],
    orders: Sequence[float],
    edit_counts: Sequence[int],
    quantiles: Sequence[float],
) -> dict[str, Any]:
    moment_rows: list[dict[str, Any]] = []
    quantile_rows: list[dict[str, Any]] = []
    variational_rows: list[dict[str, Any]] = []

    for p in probabilities:
        for rho in orders:
            alpha = 1.0 / (1.0 + rho)
            q_star = p**alpha / (p**alpha + (1.0 - p) ** alpha)
            variational = rho * h2(q_star) - binary_kl(q_star, p)
            target = rho * binary_renyi(p, alpha)
            variational_rows.append(
                {
                    "p": float(p),
                    "rho": float(rho),
                    "alpha": float(alpha),
                    "q_star": float(q_star),
                    "variational_value": float(variational),
                    "renyi_target": float(target),
                    "absolute_error": abs(float(variational - target)),
                }
            )
        for t in edit_counts:
            for n in blocklengths:
                if t >= n:
                    continue
                for rho in orders:
                    rates = exact_moment_rates(int(n), int(t), float(p), float(rho))
                    moment_rows.append(
                        {
                            "n": int(n),
                            "t": int(t),
                            "p": float(p),
                            "rho": float(rho),
                            "reveal_lower_rate": rates.reveal_lower,
                            "reveal_upper_rate": rates.reveal_upper,
                            "certificate_upper_rate": rates.certificate_upper,
                            "theory_rate": rates.theory,
                            "reveal_lower_gap": rates.theory - rates.reveal_lower,
                            "reveal_upper_gap": rates.reveal_upper - rates.theory,
                            "certificate_upper_gap": rates.certificate_upper - rates.theory,
                        }
                    )
                for quantile in quantiles:
                    quantile_rows.append(
                        {
                            "n": int(n),
                            "t": int(t),
                            "p": float(p),
                            "quantile": float(quantile),
                            "reveal_quantile_rate": reveal_quantile_rate(n, t, p, quantile, False),
                            "certificate_quantile_upper_rate": reveal_quantile_rate(n, t, p, quantile, True),
                            "typical_target_h2": h2(p),
                        }
                    )

    moment_frame = pd.DataFrame(moment_rows)
    quantile_frame = pd.DataFrame(quantile_rows)
    variational_frame = pd.DataFrame(variational_rows)
    moment_frame.to_csv(output_dir / "02_moment_rates.csv", index=False)
    quantile_frame.to_csv(output_dir / "02_tail_quantiles.csv", index=False)
    variational_frame.to_csv(output_dir / "02_variational_identity.csv", index=False)

    largest_n = max(blocklengths)
    largest = moment_frame[moment_frame["n"] == largest_n]
    largest_q = quantile_frame[quantile_frame["n"] == largest_n]
    reveal_gap = float(
        max(largest["reveal_upper_gap"].abs().max(), largest["reveal_lower_gap"].abs().max())
    )
    cert_gap = float(largest["certificate_upper_gap"].max())
    quantile_gap = float(
        (largest_q["reveal_quantile_rate"] - largest_q["typical_target_h2"]).abs().max()
    )

    decreasing_flags: list[bool] = []
    for _, group in moment_frame.groupby(["t", "p", "rho"]):
        group = group.sort_values("n")
        gaps = np.maximum(
            np.abs(group["reveal_upper_gap"].to_numpy(float)),
            np.abs(group["reveal_lower_gap"].to_numpy(float)),
        )
        decreasing_flags.append(bool(gaps[-1] <= gaps[0] + 1e-12))

    ldp_rows = []
    for p in probabilities:
        for gamma in np.linspace(max(0.0, h2(p)), 1.0, 9):
            ldp_rows.append({"p": p, "gamma": float(gamma), "rate": reveal_ldp_rate(p, float(gamma))})
        for beta in (0.0, 0.01, 0.05, 0.1):
            ldp_rows.append(
                {
                    "p": p,
                    "tail_beta": beta,
                    "tail_quantile_exponent": exponential_tail_quantile_exponent(p, beta),
                }
            )
    pd.DataFrame(ldp_rows).to_csv(output_dir / "02_ldp_phase.csv", index=False)

    payload = {
        "largest_n": int(largest_n),
        "maximum_largest_n_reveal_moment_gap": reveal_gap,
        "maximum_largest_n_certificate_upper_gap": cert_gap,
        "maximum_largest_n_quantile_gap": quantile_gap,
        "gap_decrease_fraction": float(np.mean(decreasing_flags)) if decreasing_flags else 0.0,
        "variational_identity_max_error": float(variational_frame["absolute_error"].max()),
        "theorem_statement": (
            "For fixed t, p in (0,1/2), and rho>0, probability-ordered history revelation has moment exponent "
            "rho H_{1/(1+rho)}(p). The actual certified FIBER history work obeys the corresponding upper moment "
            "and upper-tail bounds because it stops no later than the deterministic shell certificate."
        ),
        "equality_boundary": (
            "The equality is for the history-revelation variable, not for every code-dependent certified decoder. "
            "The decoder may stop earlier after finding a competing ML codeword."
        ),
    }
    write_json(output_dir / "02_moment_tail_gate.json", payload)
    return payload
