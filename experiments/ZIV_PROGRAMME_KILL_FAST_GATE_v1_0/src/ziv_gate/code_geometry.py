from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from .codelengths import ctw_codelength_bits, lz78_codelength_bits
from .processes import sample_markov
from .utils import binary_tuple, write_json


@dataclass
class LinearCode:
    n: int
    k: int
    row_masks: tuple[int, ...]
    codewords: np.ndarray

    @classmethod
    def random_systematic(cls, n: int, k: int, rng: np.random.Generator) -> "LinearCode":
        if not (1 <= k < n):
            raise ValueError("Require 1 <= k < n")
        rows: list[int] = []
        for i in range(k):
            mask = 1 << i
            parity = rng.integers(0, 2, size=n-k, dtype=np.uint8)
            for j, bit in enumerate(parity):
                if bit:
                    mask |= 1 << (k + j)
            rows.append(mask)
        size = 1 << k
        words = np.zeros(size, dtype=np.int64)
        # Gray-code enumeration: one generator row changes per step.
        previous_gray = 0
        word = 0
        for message in range(1, size):
            gray = message ^ (message >> 1)
            changed = gray ^ previous_gray
            row_index = changed.bit_length() - 1
            word ^= rows[row_index]
            words[message] = word
            previous_gray = gray
        return cls(n, k, tuple(rows), words)

    @property
    def rate(self) -> float:
        return self.k / self.n

    def minimum_hamming_distance(self) -> int:
        return min(int(word).bit_count() for word in self.codewords[1:])

    def to_dict(self) -> dict[str, Any]:
        return {"n": self.n, "k": self.k, "rate": self.rate, "row_masks": list(self.row_masks)}


def _rank_lengths(raw_lengths: np.ndarray) -> np.ndarray:
    order = np.argsort(raw_lengths, kind="mergesort")
    output = np.empty(len(raw_lengths), dtype=float)
    cursor = 0
    cumulative = 0
    while cursor < len(order):
        start = cursor
        reference = raw_lengths[order[cursor]]
        while cursor < len(order) and abs(float(raw_lengths[order[cursor]]) - float(reference)) <= 1e-12:
            cursor += 1
        cumulative += cursor - start
        output[order[start:cursor]] = math.log2(cumulative)
    return output


def ambient_metric_arrays(n: int, include_ctw: bool, ctw_depth: int) -> dict[str, dict[str, np.ndarray]]:
    ambient = 1 << n
    sequences = [binary_tuple(value, n) for value in range(ambient)]
    lz = np.fromiter((lz78_codelength_bits(bits) for bits in sequences), dtype=float, count=ambient)
    output = {"LZ78_FIXED_BLOCK": {"raw": lz, "rank": _rank_lengths(lz)}}
    if include_ctw:
        ctw = np.fromiter((ctw_codelength_bits(bits, depth=ctw_depth) for bits in sequences), dtype=float, count=ambient)
        output[f"CTW_D{ctw_depth}"] = {"raw": ctw, "rank": _rank_lengths(ctw)}
    return output


def _lfsr_int(n: int) -> int:
    state = 0b11111
    value = 0
    for i in range(n):
        if state & 1:
            value |= 1 << i
        feedback = ((state >> 0) ^ (state >> 2)) & 1
        state = (state >> 1) | (feedback << 4)
        if state == 0:
            state = 1
    return value


def structured_errors(n: int, rng: np.random.Generator) -> list[tuple[str, int, str]]:
    patterns: list[tuple[str, int, str]] = []
    patterns.append(("all_ones", (1 << n) - 1, "structured"))
    patterns.append(("alternating", sum((i & 1) << i for i in range(n)), "structured"))
    patterns.append(("period_001", sum((1 if i % 3 == 2 else 0) << i for i in range(n)), "structured"))
    patterns.append(("lfsr5", _lfsr_int(n), "structured"))
    for fraction in (0.25, 0.5, 0.75):
        length = max(1, int(round(n * fraction)))
        start = max(0, (n - length) // 2)
        mask = ((1 << length) - 1) << start
        patterns.append((f"central_burst_{fraction:.2f}", mask, "structured"))
    quarter = max(1, n // 4)
    two = ((1 << quarter) - 1) | (((1 << quarter) - 1) << max(quarter, n - quarter))
    patterns.append(("two_edge_bursts", two & ((1 << n) - 1), "structured"))
    markov = sample_markov(n, 0.02, 0.15, rng)
    markov_int = sum(bit << i for i, bit in enumerate(markov))
    patterns.append(("markov_bursty", markov_int, "structured"))
    # Matched-weight random controls for every nontrivial structured pattern.
    controls: list[tuple[str, int, str]] = []
    for label, value, _ in patterns:
        weight = int(value).bit_count()
        positions = rng.choice(n, size=weight, replace=False) if weight else []
        random_value = 0
        for pos in positions:
            random_value |= 1 << int(pos)
        controls.append((f"random_weight_matched_{label}", random_value, "random_control"))
    return patterns + controls


def _code_objective(code: LinearCode, rank_lengths: np.ndarray) -> float:
    return float(np.min(rank_lengths[code.codewords[1:]]))


def select_code(n: int, k: int, rng: np.random.Generator, candidate_pool: int, rank_lengths: np.ndarray) -> tuple[LinearCode, LinearCode, list[float]]:
    candidates = [LinearCode.random_systematic(n, k, rng) for _ in range(candidate_pool)]
    objectives = [_code_objective(code, rank_lengths) for code in candidates]
    best = candidates[int(np.argmax(objectives))]
    return candidates[0], best, objectives


def _evaluate_code(
    code: LinearCode,
    label: str,
    metric_name: str,
    raw_lengths: np.ndarray,
    rank_lengths: np.ndarray,
    errors: Sequence[tuple[str, int, str]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    nonzero = code.codewords[1:]
    raw_nonzero = raw_lengths[nonzero]
    rank_nonzero = rank_lengths[nonzero]
    d_h = code.minimum_hamming_distance()
    t_h = (d_h - 1) // 2
    code_summary = {
        "code_label": label,
        "metric": metric_name,
        "n": code.n,
        "k": code.k,
        "rate": code.rate,
        "minimum_hamming_distance": d_h,
        "hamming_unique_radius": t_h,
        "minimum_raw_description_length_nonzero": float(np.min(raw_nonzero)),
        "minimum_rank_description_bits_nonzero": float(np.min(rank_nonzero)),
        "minimum_rank_description_fraction_of_redundancy": float(np.min(rank_nonzero) / max(1, code.n-code.k)),
    }
    rows: list[dict[str, Any]] = []
    for error_label, z, family in errors:
        lz = float(raw_lengths[z])
        competitors = np.bitwise_xor(nonzero, int(z))
        best_competing = float(np.min(raw_lengths[competitors]))
        mdl_success = bool(lz < best_competing - 1e-12)
        weight = int(z).bit_count()
        hamming_guaranteed = bool(weight <= t_h)
        rows.append(
            {
                "code_label": label,
                "metric": metric_name,
                "n": code.n,
                "k": code.k,
                "rate": code.rate,
                "error_label": error_label,
                "error_family": family,
                "error_weight": weight,
                "relative_weight": weight / code.n,
                "true_error_codelength": lz,
                "best_competing_error_codelength": best_competing,
                "codelength_margin": best_competing - lz,
                "mdl_unique_correction": mdl_success,
                "hamming_guaranteed": hamming_guaranteed,
                "mdl_only_success": bool(mdl_success and not hamming_guaranteed),
            }
        )
    return code_summary, rows



def lz_asymptotic_crossover(rates: Sequence[float]) -> dict[str, Any]:
    lengths = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144]
    patterns = {
        "constant_zero": lambda n: (0,) * n,
        "alternating": lambda n: tuple(i & 1 for i in range(n)),
        "period_001": lambda n: tuple(1 if i % 3 == 2 else 0 for i in range(n)),
    }
    output: dict[str, Any] = {}
    for rate in rates:
        rate_key = f"R={rate}"
        output[rate_key] = {}
        threshold_fraction = (1.0-rate)/2.0
        for label, generator in patterns.items():
            rows = []
            first = None
            for n in lengths:
                length = lz78_codelength_bits(generator(n))
                threshold = threshold_fraction*n
                rows.append({"n": n, "lz_bits": length, "half_redundancy_bits": threshold, "passes": length < threshold})
                if first is None and length < threshold:
                    first = n
            output[rate_key][label] = {"first_tested_crossover_n": first, "rows": rows}
    return output

def run_code_geometry_gate(
    output_dir: Path,
    rng: np.random.Generator,
    blocklengths: Sequence[int],
    rates: Sequence[float],
    replicates: int,
    candidate_pool: int,
    ctw_depth: int,
    ctw_max_n: int,
) -> dict[str, Any]:
    code_rows: list[dict[str, Any]] = []
    error_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    for n in blocklengths:
        metric_arrays = ambient_metric_arrays(n, include_ctw=n <= ctw_max_n, ctw_depth=ctw_depth)
        errors = structured_errors(n, rng)
        for rate in rates:
            k = max(1, min(n-1, int(round(rate*n))))
            for replicate in range(replicates):
                baseline, optimized, objectives = select_code(
                    n, k, rng, candidate_pool, metric_arrays["LZ78_FIXED_BLOCK"]["rank"]
                )
                selection_rows.append(
                    {
                        "n": n,
                        "k": k,
                        "rate": k/n,
                        "replicate": replicate,
                        "candidate_pool": candidate_pool,
                        "baseline_lz_rank_distance": objectives[0],
                        "best_lz_rank_distance": max(objectives),
                        "selection_gain_bits": max(objectives) - objectives[0],
                    }
                )
                for code_label, code in (("RANDOM_BASELINE", baseline), ("LZ_SELECTED", optimized)):
                    for metric_name, arrays in metric_arrays.items():
                        summary, details = _evaluate_code(
                            code,
                            code_label,
                            metric_name,
                            arrays["raw"],
                            arrays["rank"],
                            errors,
                        )
                        summary["replicate"] = replicate
                        code_rows.append(summary)
                        for row in details:
                            row["replicate"] = replicate
                            error_rows.append(row)
    code_frame = pd.DataFrame(code_rows)
    error_frame = pd.DataFrame(error_rows)
    selection_frame = pd.DataFrame(selection_rows)
    code_frame.to_csv(output_dir / "06_code_geometry_codes.csv", index=False)
    error_frame.to_csv(output_dir / "06_code_geometry_errors.csv.gz", index=False, compression="gzip")
    selection_frame.to_csv(output_dir / "06_code_geometry_selection.csv", index=False)

    largest_n = max(blocklengths)
    largest_rate = max(rates)
    details: dict[str, Any] = {}
    for metric in sorted(error_frame["metric"].unique()):
        subset = error_frame[
            (error_frame["metric"] == metric)
            & (error_frame["n"] == largest_n if metric == "LZ78_FIXED_BLOCK" else error_frame["n"] == min(largest_n, ctw_max_n))
        ]
        if subset.empty:
            continue
        metric_n = int(subset["n"].iloc[0])
        subset = subset[subset["rate"] == subset["rate"].max()]
        metric_detail: dict[str, Any] = {"n": metric_n, "rate": float(subset["rate"].max())}
        for code_label in ("RANDOM_BASELINE", "LZ_SELECTED"):
            group = subset[subset["code_label"] == code_label]
            structured = group[group["error_family"] == "structured"]
            controls = group[group["error_family"] == "random_control"]
            metric_detail[code_label] = {
                "structured_mdl_correction_fraction": float(structured["mdl_unique_correction"].mean()),
                "structured_mdl_only_fraction": float(structured["mdl_only_success"].mean()),
                "random_control_mdl_correction_fraction": float(controls["mdl_unique_correction"].mean()),
                "mean_structured_codelength_margin": float(structured["codelength_margin"].mean()),
            }
        details[metric] = metric_detail

    lz = details.get("LZ78_FIXED_BLOCK", {})
    selected = lz.get("LZ_SELECTED", {})
    baseline = lz.get("RANDOM_BASELINE", {})
    positive_signal = bool(
        selected
        and selected["structured_mdl_only_fraction"] >= 0.50
        and selected["structured_mdl_correction_fraction"]
        >= baseline.get("structured_mdl_correction_fraction", 0.0)
        and selected["structured_mdl_correction_fraction"]
        - selected["random_control_mdl_correction_fraction"] >= 0.25
    )
    payload = {
        "largest_n": largest_n,
        "lz_half_redundancy_crossover": lz_asymptotic_crossover(rates),
        "largest_rate_requested": largest_rate,
        "candidate_pool": candidate_pool,
        "details": details,
        "selection_mean_gain_bits": float(selection_frame["selection_gain_bits"].mean()),
        "positive_structured_code_geometry_signal": positive_signal,
        "pass": positive_signal,
        "interpretation": (
            "A pass is only evidence that the frozen practical codelength induces a useful finite-block ordering. "
            "It does not prove the masking theorem, capacity, or novelty of description-length code design."
        ),
    }
    write_json(output_dir / "06_code_geometry_gate.json", payload)
    return payload
