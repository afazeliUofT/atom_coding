from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from .channels import FixedDeletionBSC, sample_channel
from .codes import VTOracle, make_linear
from .history_decoder import history_decode
from .phase_diagram import threshold_for_target
from .prefix_astar import prefix_aggregate_astar
from .syndrome_trellis import syndrome_trellis_aggregate_decode
from .utils import safe_ratio, slope_log2, write_json
from .vt_linear import vt_decode_single_deletion_linear


def _trial_count(n: int, schedule: dict[str, int], default: int = 4) -> int:
    return int(schedule.get(str(n), default))


def run_primary_benchmark(
    output_dir: Path,
    rng: np.random.Generator,
    blocklengths: Sequence[int],
    rates: Sequence[float],
    probabilities: Sequence[float],
    families: Sequence[str],
    trial_schedule: dict[str, int],
    max_histories: int,
    max_trellis_terminals: int,
    max_prefix_nodes: int,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    metadata: list[dict[str, Any]] = []

    for family in families:
        for rate in rates:
            for n in blocklengths:
                k = max(1, min(n - 1, int(round(float(rate) * n))))
                code = make_linear(family, n, k, rng, label="_FINAL")
                metadata.append(code.metadata())
                for p in probabilities:
                    channel = FixedDeletionBSC(n, 1, float(p))
                    print(f"    benchmark family={code.family} R={code.rate:.3f} n={n} p={float(p):.3f}", flush=True)
                    for trial in range(_trial_count(n, trial_schedule)):
                        word = code.sample_codeword(rng)
                        received, deleted, error = sample_channel(word, channel, rng)
                        order = list(rng.permutation(3))
                        outcomes: dict[str, Any] = {}
                        for which in order:
                            if which == 0:
                                outcomes["fiber"] = history_decode(received, channel, code, max_histories)
                            elif which == 1:
                                outcomes["trellis"] = syndrome_trellis_aggregate_decode(
                                    received, channel, code, max_trellis_terminals
                                )
                            else:
                                outcomes["prefix"] = prefix_aggregate_astar(
                                    received, channel, code, max_prefix_nodes
                                )
                        fiber = outcomes["fiber"]
                        trellis = outcomes["trellis"]
                        prefix = outcomes["prefix"]
                        agreement = bool(
                            fiber.certified
                            and trellis.certified
                            and prefix.certified
                            and set(fiber.tie_words) == set(trellis.tie_words) == set(prefix.tie_words)
                        )
                        best_wall = min(trellis.work.wall_seconds, prefix.work.wall_seconds)
                        best_work = min(trellis.work.scalar(), prefix.work.scalar())
                        redundancy = 1.0 - code.rate
                        typical_threshold = threshold_for_target(redundancy, "typical")
                        mean_threshold = threshold_for_target(redundancy, "mean")
                        rows.append(
                            {
                                "family": code.family,
                                "code_name": code.name,
                                "n": n,
                                "k": code.k,
                                "rate": code.rate,
                                "p": float(p),
                                "trial": trial,
                                "deleted_position": deleted[0],
                                "error_weight": int(error).bit_count(),
                                "agreement": agreement,
                                "fiber_certified": fiber.certified,
                                "trellis_certified": trellis.certified,
                                "prefix_certified": prefix.certified,
                                "fiber_wall": fiber.work.wall_seconds,
                                "trellis_wall": trellis.work.wall_seconds,
                                "prefix_wall": prefix.work.wall_seconds,
                                "best_baseline_wall": best_wall,
                                "fiber_over_best_wall": safe_ratio(fiber.work.wall_seconds, best_wall),
                                "fiber_work": fiber.work.scalar(),
                                "trellis_work": trellis.work.scalar(),
                                "prefix_work": prefix.work.scalar(),
                                "best_baseline_work": best_work,
                                "fiber_over_best_work": safe_ratio(fiber.work.scalar(), best_work),
                                "fiber_histories": fiber.work.histories,
                                "fiber_membership_queries": fiber.work.membership_queries,
                                "fiber_exact_scores": fiber.work.exact_score_calls,
                                "fiber_peak_seen": fiber.work.peak_seen,
                                "trellis_dp_updates": trellis.work.trellis_dp_updates,
                                "trellis_nodes": trellis.work.trellis_nodes,
                                "prefix_nodes": prefix.work.trellis_nodes,
                                "typical_threshold_p": typical_threshold,
                                "mean_threshold_p": mean_threshold,
                                "predicted_typical_favorable": float(p) < typical_threshold,
                                "predicted_mean_favorable": float(p) < mean_threshold,
                            }
                        )
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "04_primary_trials.csv.gz", index=False, compression="gzip")

    summary_rows: list[dict[str, Any]] = []
    for (family, rate, p, n), group in frame.groupby(["family", "rate", "p", "n"]):
        summary_rows.append(
            {
                "family": family,
                "rate": float(rate),
                "p": float(p),
                "n": int(n),
                "trials": len(group),
                "agreement_fraction": float(group["agreement"].mean()),
                "completion_fraction": float(
                    (group["fiber_certified"] & group["trellis_certified"] & group["prefix_certified"]).mean()
                ),
                "predicted_typical_favorable": bool(group["predicted_typical_favorable"].iloc[0]),
                "predicted_mean_favorable": bool(group["predicted_mean_favorable"].iloc[0]),
                "median_fiber_over_best_wall": float(group["fiber_over_best_wall"].median()),
                "p95_fiber_over_best_wall": float(group["fiber_over_best_wall"].quantile(0.95)),
                "maximum_fiber_over_best_wall": float(group["fiber_over_best_wall"].max()),
                "median_fiber_over_best_work": float(group["fiber_over_best_work"].median()),
                "median_fiber_wall": float(group["fiber_wall"].median()),
                "median_best_baseline_wall": float(group["best_baseline_wall"].median()),
                "median_histories": float(group["fiber_histories"].median()),
                "p95_histories": float(group["fiber_histories"].quantile(0.95)),
                "median_membership_queries": float(group["fiber_membership_queries"].median()),
                "median_peak_seen": float(group["fiber_peak_seen"].median()),
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_dir / "04_primary_summary.csv", index=False)

    slope_rows = []
    for (family, rate, p), group in summary.groupby(["family", "rate", "p"]):
        group = group.sort_values("n")
        slope_rows.append(
            {
                "family": family,
                "rate": float(rate),
                "p": float(p),
                "predicted_typical_favorable": bool(group["predicted_typical_favorable"].iloc[0]),
                "fiber_wall_log2_slope": slope_log2(group["n"], group["median_fiber_wall"]),
                "best_wall_log2_slope": slope_log2(group["n"], group["median_best_baseline_wall"]),
                "fiber_minus_best_slope": slope_log2(group["n"], group["median_fiber_wall"])
                - slope_log2(group["n"], group["median_best_baseline_wall"]),
            }
        )
    slopes = pd.DataFrame(slope_rows)
    slopes.to_csv(output_dir / "04_primary_slopes.csv", index=False)

    payload = {
        "exact_agreement_all": bool(frame["agreement"].all()),
        "completion_all": bool((frame["fiber_certified"] & frame["trellis_certified"] & frame["prefix_certified"]).all()),
        "codebook_free": all(meta.get("membership") == "syndrome_only_no_codebook_table" for meta in metadata),
        "code_metadata": metadata,
        "largest_n": int(max(blocklengths)),
        "summary": summary_rows,
        "slopes": slope_rows,
    }
    write_json(output_dir / "04_primary_benchmark_gate.json", payload)
    return payload


def run_vt_boundary_gate(
    output_dir: Path,
    rng: np.random.Generator,
    blocklengths: Sequence[int],
    trials: int,
) -> dict[str, Any]:
    rows = []
    for n in blocklengths:
        code = VTOracle(int(n), 0)
        channel = FixedDeletionBSC(int(n), 1, 0.0)
        for trial in range(trials):
            word = code.sample_codeword(rng)
            received, _, _ = sample_channel(word, channel, rng)
            fiber = history_decode(received, channel, code, max_histories=8 * n)
            direct = vt_decode_single_deletion_linear(received, code)
            rows.append(
                {
                    "n": int(n),
                    "trial": trial,
                    "agreement": fiber.certified and direct.valid and fiber.decision_word == direct.word == word,
                    "fiber_wall": fiber.work.wall_seconds,
                    "vt_linear_wall": direct.work.wall_seconds,
                    "fiber_over_vt_wall": safe_ratio(fiber.work.wall_seconds, direct.work.wall_seconds),
                    "fiber_histories": fiber.work.histories,
                    "vt_candidate_checks": direct.work.vt_candidate_checks,
                }
            )
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "05_vt_boundary_trials.csv", index=False)
    summary_rows = []
    for n, group in frame.groupby("n"):
        summary_rows.append(
            {
                "n": int(n),
                "trials": len(group),
                "agreement_fraction": float(group["agreement"].mean()),
                "median_fiber_over_vt_wall": float(group["fiber_over_vt_wall"].median()),
                "p95_fiber_over_vt_wall": float(group["fiber_over_vt_wall"].quantile(0.95)),
                "median_fiber_histories": float(group["fiber_histories"].median()),
                "median_vt_candidate_checks": float(group["vt_candidate_checks"].median()),
            }
        )
    pd.DataFrame(summary_rows).to_csv(output_dir / "05_vt_boundary_summary.csv", index=False)
    payload = {
        "agreement_all": bool(frame["agreement"].all()),
        "linear_candidate_check_identity": bool(
            all(row["median_vt_candidate_checks"] <= row["n"] + 1 for row in summary_rows)
        ),
        "summary": summary_rows,
        "interpretation": (
            "The classical VT decoder is a specialization boundary, not a target FIBER should beat. "
            "Its O(n) reconstruction demonstrates the value of code-specific structure."
        ),
    }
    write_json(output_dir / "05_vt_boundary_gate.json", payload)
    return payload
