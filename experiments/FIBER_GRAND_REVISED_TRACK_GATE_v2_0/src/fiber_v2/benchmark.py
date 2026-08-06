from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from .channels import FixedDeletionBSC, sample_channel
from .codes import LinearOracle, VTOracle, make_linear
from .history_decoder import history_decode
from .shell_theory import certificate_inequality_holds, shell_certificate_bound
from .prefix_astar import prefix_aggregate_astar
from .syndrome_trellis import syndrome_trellis_aggregate_decode
from .utils import percentile, safe_ratio, slope_log2, write_json
from .vt_baseline import vt_direct_one_deletion


def _trial_count(n: int, schedule: dict[str, int], default: int = 4) -> int:
    return int(schedule.get(str(n), default))


def run_primary_strong_baseline(
    output_dir: Path,
    rng: np.random.Generator,
    blocklengths: Sequence[int],
    rate: float,
    probabilities: Sequence[float],
    families: Sequence[str],
    trial_schedule: dict[str, int],
    max_histories: int,
    max_trellis_terminals: int,
    max_prefix_nodes: int,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    code_metadata: list[dict[str, Any]] = []
    for family in families:
        for n in blocklengths:
            k = max(1, min(n - 1, int(round(rate * n))))
            code = make_linear(family, n, k, rng, label="_V2")
            code_metadata.append(code.metadata())
            for p in probabilities:
                channel = FixedDeletionBSC(n, 1, float(p))
                for trial in range(_trial_count(n, trial_schedule)):
                    transmitted = code.sample_codeword(rng)
                    received, deleted, error = sample_channel(transmitted, channel, rng)
                    fiber = history_decode(received, channel, code, max_histories=max_histories)
                    trellis = syndrome_trellis_aggregate_decode(
                        received, channel, code, max_terminals=max_trellis_terminals
                    )
                    prefix = prefix_aggregate_astar(received, channel, code, max_nodes=max_prefix_nodes)
                    agreement = bool(
                        fiber.certified
                        and trellis.certified
                        and prefix.certified
                        and fiber.decision_word is not None
                        and trellis.decision_word is not None
                        and prefix.decision_word is not None
                        and set(fiber.tie_words) == set(trellis.tie_words) == set(prefix.tie_words)
                    )
                    best_baseline_wall = min(trellis.work.wall_seconds, prefix.work.wall_seconds)
                    best_baseline_name = "SYNDROME_TRELLIS" if trellis.work.wall_seconds <= prefix.work.wall_seconds else "PREFIX_ASTAR"
                    best_baseline_work = min(trellis.work.scalar(), prefix.work.scalar())
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
                            "best_baseline_wall": best_baseline_wall,
                            "best_baseline_name": best_baseline_name,
                            "fiber_over_trellis_wall": safe_ratio(fiber.work.wall_seconds, trellis.work.wall_seconds),
                            "fiber_over_prefix_wall": safe_ratio(fiber.work.wall_seconds, prefix.work.wall_seconds),
                            "fiber_over_best_wall": safe_ratio(fiber.work.wall_seconds, best_baseline_wall),
                            "best_over_fiber_wall": safe_ratio(best_baseline_wall, fiber.work.wall_seconds),
                            "fiber_work": fiber.work.scalar(),
                            "trellis_work": trellis.work.scalar(),
                            "prefix_work": prefix.work.scalar(),
                            "best_baseline_work": best_baseline_work,
                            "fiber_over_trellis_work": safe_ratio(fiber.work.scalar(), trellis.work.scalar()),
                            "fiber_histories": fiber.work.histories,
                            "fiber_distinct_candidates": fiber.work.distinct_candidates,
                            "fiber_membership_queries": fiber.work.membership_queries,
                            "fiber_exact_scores": fiber.work.exact_score_calls,
                            "fiber_shell_at_stop": fiber.shell_at_stop,
                            "fiber_peak_seen": fiber.work.peak_seen,
                            "trellis_dp_updates": trellis.work.trellis_dp_updates,
                            "trellis_nodes": trellis.work.trellis_nodes,
                            "trellis_terminals": trellis.work.trellis_terminals,
                            "trellis_exact_scores": trellis.work.exact_score_calls,
                            "prefix_nodes": prefix.work.trellis_nodes,
                            "prefix_exact_scores": prefix.work.exact_score_calls,
                        }
                    )
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "02_primary_trials.csv.gz", index=False, compression="gzip")
    summary_rows: list[dict[str, Any]] = []
    for (family, p, n), group in frame.groupby(["family", "p", "n"]):
        summary_rows.append(
            {
                "family": family,
                "p": float(p),
                "n": int(n),
                "trials": len(group),
                "agreement_fraction": float(group["agreement"].mean()),
                "fiber_completion_fraction": float(group["fiber_certified"].mean()),
                "trellis_completion_fraction": float(group["trellis_certified"].mean()),
                "prefix_completion_fraction": float(group["prefix_certified"].mean()),
                "median_fiber_wall": float(group["fiber_wall"].median()),
                "median_trellis_wall": float(group["trellis_wall"].median()),
                "median_prefix_wall": float(group["prefix_wall"].median()),
                "median_best_baseline_wall": float(group["best_baseline_wall"].median()),
                "median_fiber_over_trellis_wall": float(group["fiber_over_trellis_wall"].median()),
                "median_fiber_over_prefix_wall": float(group["fiber_over_prefix_wall"].median()),
                "median_fiber_over_best_wall": float(group["fiber_over_best_wall"].median()),
                "p95_fiber_over_best_wall": float(group["fiber_over_best_wall"].quantile(0.95)),
                "median_best_over_fiber_wall": float(group["best_over_fiber_wall"].median()),
                "median_fiber_work": float(group["fiber_work"].median()),
                "median_trellis_work": float(group["trellis_work"].median()),
                "median_prefix_work": float(group["prefix_work"].median()),
                "median_best_baseline_work": float(group["best_baseline_work"].median()),
                "median_fiber_histories": float(group["fiber_histories"].median()),
                "p99_fiber_histories": float(group["fiber_histories"].quantile(0.99)),
                "median_fiber_membership_queries": float(group["fiber_membership_queries"].median()),
                "median_trellis_dp_updates": float(group["trellis_dp_updates"].median()),
                "median_trellis_nodes": float(group["trellis_nodes"].median()),
                "median_prefix_nodes": float(group["prefix_nodes"].median()),
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_dir / "02_primary_summary.csv", index=False)

    slope_rows: list[dict[str, Any]] = []
    for (family, p), group in summary.groupby(["family", "p"]):
        group = group.sort_values("n")
        slope_rows.append(
            {
                "family": family,
                "p": float(p),
                "fiber_wall_log2_slope": slope_log2(group["n"], group["median_fiber_wall"]),
                "trellis_wall_log2_slope": slope_log2(group["n"], group["median_trellis_wall"]),
                "prefix_wall_log2_slope": slope_log2(group["n"], group["median_prefix_wall"]),
                "best_baseline_wall_log2_slope": slope_log2(group["n"], group["median_best_baseline_wall"]),
                "fiber_work_log2_slope": slope_log2(group["n"], group["median_fiber_work"]),
                "trellis_work_log2_slope": slope_log2(group["n"], group["median_trellis_work"]),
                "prefix_work_log2_slope": slope_log2(group["n"], group["median_prefix_work"]),
            }
        )
    slopes = pd.DataFrame(slope_rows)
    slopes.to_csv(output_dir / "02_primary_slopes.csv", index=False)
    payload = {
        "exact_agreement_all": bool(frame["agreement"].all()),
        "codebook_free": all(meta.get("membership") == "syndrome_only_no_codebook_table" for meta in code_metadata),
        "code_metadata": code_metadata,
        "largest_n": max(blocklengths),
        "summary": summary_rows,
        "slopes": slope_rows,
    }
    write_json(output_dir / "02_primary_gate.json", payload)
    return payload


def run_two_deletion_diagnostic(
    output_dir: Path,
    rng: np.random.Generator,
    blocklengths: Sequence[int],
    rate: float,
    probabilities: Sequence[float],
    trial_schedule: dict[str, int],
    max_histories: int,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for n in blocklengths:
        k = max(1, min(n - 1, int(round(rate * n))))
        code = make_linear("RLC", n, k, rng, label="_D2_V2")
        for p in probabilities:
            channel = FixedDeletionBSC(n, 2, float(p))
            for trial in range(_trial_count(n, trial_schedule, 2)):
                transmitted = code.sample_codeword(rng)
                received, deleted, error = sample_channel(transmitted, channel, rng)
                outcome = history_decode(received, channel, code, max_histories=max_histories)
                bound = shell_certificate_bound(channel, int(error).bit_count())
                rows.append(
                    {
                        "n": n,
                        "k": k,
                        "rate": code.rate,
                        "p": float(p),
                        "trial": trial,
                        "certified": outcome.certified,
                        "error_weight": int(error).bit_count(),
                        "histories": outcome.work.histories,
                        "membership_queries": outcome.work.membership_queries,
                        "membership_fraction": outcome.work.membership_queries / (1 << k),
                        "wall": outcome.work.wall_seconds,
                        "theorem_history_bound": bound.history_upper_bound,
                        "observed_below_theorem_bound": outcome.work.histories <= bound.history_upper_bound,
                    }
                )
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "03_two_deletion_trials.csv", index=False)
    summary_rows = []
    for (p, n), group in frame.groupby(["p", "n"]):
        summary_rows.append(
            {
                "p": float(p),
                "n": int(n),
                "trials": len(group),
                "completion_fraction": float(group["certified"].mean()),
                "median_histories": float(group["histories"].median()),
                "p95_histories": float(group["histories"].quantile(0.95)),
                "median_membership_fraction": float(group["membership_fraction"].median()),
                "median_wall": float(group["wall"].median()),
                "all_below_theorem_bound": bool(group["observed_below_theorem_bound"].all()),
            }
        )
    pd.DataFrame(summary_rows).to_csv(output_dir / "03_two_deletion_summary.csv", index=False)
    payload = {
        "completion_all": bool(frame["certified"].all()),
        "all_observed_below_theorem_bound": bool(frame["observed_below_theorem_bound"].all()),
        "summary": summary_rows,
    }
    write_json(output_dir / "03_two_deletion_gate.json", payload)
    return payload


def run_vt_specialized_gate(
    output_dir: Path,
    rng: np.random.Generator,
    blocklengths: Sequence[int],
    trials: int,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for n in blocklengths:
        code = VTOracle(int(n), 0)
        channel = FixedDeletionBSC(int(n), 1, 0.0)
        for trial in range(trials):
            transmitted = code.sample_codeword(rng)
            received, _, _ = sample_channel(transmitted, channel, rng)
            fiber = history_decode(received, channel, code, max_histories=10 * n)
            direct = vt_direct_one_deletion(received, code)
            rows.append(
                {
                    "n": n,
                    "trial": trial,
                    "agreement": fiber.certified and set(fiber.tie_words) == set(direct.tie_words),
                    "fiber_wall": fiber.work.wall_seconds,
                    "vt_wall": direct.work.wall_seconds,
                    "fiber_over_vt_wall": safe_ratio(fiber.work.wall_seconds, direct.work.wall_seconds),
                    "fiber_histories": fiber.work.histories,
                    "fiber_membership_queries": fiber.work.membership_queries,
                    "vt_candidate_checks": direct.work.vt_candidate_checks,
                }
            )
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "04_vt_trials.csv", index=False)
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
    pd.DataFrame(summary_rows).to_csv(output_dir / "04_vt_summary.csv", index=False)
    payload = {"agreement_all": bool(frame["agreement"].all()), "summary": summary_rows}
    write_json(output_dir / "04_vt_gate.json", payload)
    return payload


def run_shell_theorem_gate(
    output_dir: Path,
    rng: np.random.Generator,
    blocklengths: Sequence[int],
    probabilities: Sequence[float],
    samples: int,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    all_inequalities = True
    for n in blocklengths:
        for t in (1, 2):
            for p in probabilities:
                channel = FixedDeletionBSC(int(n), t, float(p))
                for sample in range(samples):
                    e = int(rng.binomial(channel.m, channel.p))
                    bound = shell_certificate_bound(channel, e)
                    inequality = certificate_inequality_holds(channel, e)
                    all_inequalities &= inequality
                    rows.append(
                        {
                            "n": n,
                            "t": t,
                            "p": float(p),
                            "sample": sample,
                            "error_weight": e,
                            "extra_shells": bound.extra_shells,
                            "certificate_shell": bound.certificate_shell,
                            "log2_history_bound_per_n": bound.normalized_log2_bound,
                            "h2_p": bound.asymptotic_h2_p,
                            "excess_over_h2": bound.normalized_log2_bound - bound.asymptotic_h2_p,
                            "inequality_holds": inequality,
                        }
                    )
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "05_shell_theorem_trials.csv.gz", index=False, compression="gzip")
    summary_rows = []
    for (t, p, n), group in frame.groupby(["t", "p", "n"]):
        summary_rows.append(
            {
                "t": int(t),
                "p": float(p),
                "n": int(n),
                "median_log2_bound_per_n": float(group["log2_history_bound_per_n"].median()),
                "p95_log2_bound_per_n": float(group["log2_history_bound_per_n"].quantile(0.95)),
                "median_excess_over_h2": float(group["excess_over_h2"].median()),
                "p95_excess_over_h2": float(group["excess_over_h2"].quantile(0.95)),
                "all_inequalities": bool(group["inequality_holds"].all()),
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_dir / "05_shell_theorem_summary.csv", index=False)
    largest_n = max(blocklengths)
    large = summary[summary["n"] == largest_n]
    payload = {
        "all_inequalities": bool(all_inequalities),
        "largest_n": largest_n,
        "maximum_p95_excess_over_h2_at_largest_n": float(large["p95_excess_over_h2"].max()),
        "summary": summary_rows,
    }
    write_json(output_dir / "05_shell_theorem_gate.json", payload)
    return payload
