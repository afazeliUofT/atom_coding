from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from .baselines import exhaustive_deletion_ml
from .channels import DeletionChannel, sample_deletion_channel
from .experiments_common import (
    deletion_history_components,
    effective_history_count,
    history_entropy_bits,
    make_code,
    result_row_base,
)
from .history_decoder import deletion_history_model, history_fiber_decode, pathwise_first_hit
from .prefix_decoder import prefix_fiber_decode
from .utils import bootstrap_log2_slope, percentile, safe_mean, safe_median, write_json


def _trial_counts(blocklength: int, schedule: dict[str, int], default: int) -> int:
    return int(schedule.get(str(blocklength), default))


def run_one_deletion_scaling(
    output_dir: Path,
    rng: np.random.Generator,
    blocklengths: Sequence[int],
    target_rate: float,
    substitution_probabilities: Sequence[float],
    code_families: Sequence[str],
    trial_schedule: dict[str, int],
    max_histories: int,
    max_prefix_nodes: int,
    bootstrap_replicates: int,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    path_rows: list[dict[str, Any]] = []
    code_metadata: list[dict[str, Any]] = []

    for family in code_families:
        for n in blocklengths:
            k = max(1, min(n - 1, int(round(target_rate * n))))
            code = make_code(family, n, k, rng, label="_D1")
            code_metadata.append(code.metadata())
            trials = _trial_counts(n, trial_schedule, 20)
            for p in substitution_probabilities:
                channel = DeletionChannel(n=n, deletions=1, substitution_probability=float(p))
                for trial in range(trials):
                    true_message = int(rng.integers(0, code.size))
                    transmitted = int(code.codewords_int[true_message])
                    received, deleted, substitution_mask = sample_deletion_channel(transmitted, channel, rng)
                    reference = exhaustive_deletion_ml(code, received, channel)
                    reference_scalar = reference.work.scalar("balanced")
                    reference_pessimistic = reference.work.scalar("pessimistic")

                    algorithms = {
                        "HISTORY_L0": history_fiber_decode(
                            deletion_history_model(received, channel),
                            code,
                            reference.tie_set,
                            reference.decision,
                            reference.work,
                            max_histories=max_histories,
                        ),
                        "PREFIX_L0": prefix_fiber_decode(
                            received,
                            channel,
                            code,
                            reference.tie_set,
                            reference.decision,
                            reference.work,
                            max_nodes=max_prefix_nodes,
                            use_prefix_feasibility=False,
                        ),
                        "PREFIX_L2": prefix_fiber_decode(
                            received,
                            channel,
                            code,
                            reference.tie_set,
                            reference.decision,
                            reference.work,
                            max_nodes=max_prefix_nodes,
                            use_prefix_feasibility=True,
                        ),
                    }

                    path_decision, path_work = pathwise_first_hit(
                        deletion_history_model(received, channel),
                        code,
                        max_histories=max_histories,
                    )
                    path_rows.append(
                        {
                            **result_row_base(code, channel.label, n, trial, true_message, reference.tie_set),
                            "p": p,
                            "path_decision": path_decision,
                            "path_exact_ml": path_decision in reference.tie_set if path_decision is not None else False,
                            "path_histories": path_work.history_components,
                        }
                    )

                    components = deletion_history_components(transmitted, received, channel)
                    base = {
                        **result_row_base(code, channel.label, n, trial, true_message, reference.tie_set),
                        "p": p,
                        "deleted_positions": ";".join(str(v) for v in deleted),
                        "substitution_weight": int(substitution_mask).bit_count(),
                        "reference_work_balanced": reference_scalar,
                        "reference_work_pessimistic": reference_pessimistic,
                        "reference_wall_seconds": reference.work.wall_seconds,
                        "true_effective_history_count": effective_history_count(components),
                        "true_history_entropy_bits": history_entropy_bits(components),
                    }
                    for algorithm, result in algorithms.items():
                        work = result.work
                        balanced = work.scalar("balanced")
                        pessimistic = work.scalar("pessimistic")
                        rows.append(
                            {
                                **base,
                                "algorithm": algorithm,
                                "decision": result.decision,
                                "exact": result.exact,
                                "certified": result.certified,
                                "fallback": result.fallback_used,
                                "work_balanced": balanced,
                                "work_pessimistic": pessimistic,
                                "speedup_balanced": reference_scalar / max(balanced, 1e-12),
                                "speedup_pessimistic": reference_pessimistic / max(pessimistic, 1e-12),
                                "candidate_score_speedup": code.size / max(1, work.exact_score_calls),
                                "membership_fraction_of_codebook": work.membership_queries / code.size,
                                "history_components": work.history_components,
                                "distinct_candidates": work.distinct_candidates,
                                "duplicate_histories": work.duplicate_histories,
                                "membership_queries": work.membership_queries,
                                "exact_score_calls": work.exact_score_calls,
                                "prefix_nodes": work.prefix_nodes,
                                "terminal_candidates": work.terminal_candidates,
                                "peak_frontier": work.peak_frontier,
                                "wall_seconds": work.wall_seconds,
                                "incumbent_score": result.incumbent_score,
                                "residual_bound": result.residual_bound,
                            }
                        )

    frame = pd.DataFrame(rows)
    path_frame = pd.DataFrame(path_rows)
    frame.to_csv(output_dir / "04_one_deletion_trials.csv.gz", index=False, compression="gzip")
    path_frame.to_csv(output_dir / "04_one_deletion_pathwise.csv.gz", index=False, compression="gzip")

    summary_rows: list[dict[str, Any]] = []
    detailed: dict[str, Any] = {}
    for (family, p, algorithm), group in frame.groupby(["code_family", "p", "algorithm"]):
        per_n: list[dict[str, Any]] = []
        xs: list[float] = []
        work_groups: list[np.ndarray] = []
        reference_groups: list[np.ndarray] = []
        for n, ng in group.groupby("n"):
            xs.append(float(n))
            work_values = ng["work_balanced"].to_numpy(float)
            reference_values = ng["reference_work_balanced"].to_numpy(float)
            work_groups.append(work_values)
            reference_groups.append(reference_values)
            record = {
                "code_family": family,
                "p": float(p),
                "algorithm": algorithm,
                "n": int(n),
                "code_size": int(ng["code_size"].iloc[0]),
                "code_rate": float(ng["code_rate"].iloc[0]),
                "mean_work": float(np.mean(work_values)),
                "median_work": float(np.median(work_values)),
                "p95_work": float(np.percentile(work_values, 95)),
                "p99_work": float(np.percentile(work_values, 99)),
                "mean_reference_work": float(np.mean(reference_values)),
                "mean_speedup": float(np.mean(ng["speedup_balanced"])),
                "median_speedup": float(np.median(ng["speedup_balanced"])),
                "p05_speedup": float(np.percentile(ng["speedup_balanced"], 5)),
                "completion_fraction": float(np.mean(~ng["fallback"])),
                "exact_fraction": float(np.mean(ng["exact"])),
                "median_candidate_score_speedup": float(np.median(ng["candidate_score_speedup"])),
                "median_membership_fraction": float(np.median(ng["membership_fraction_of_codebook"])),
                "p99_membership_fraction": float(np.percentile(ng["membership_fraction_of_codebook"], 99)),
                "median_histories": float(np.median(ng["history_components"])),
                "p99_histories": float(np.percentile(ng["history_components"], 99)),
                "median_prefix_nodes": float(np.median(ng["prefix_nodes"])),
                "mean_wall_seconds": float(np.mean(ng["wall_seconds"])),
                "mean_reference_wall_seconds": float(np.mean(ng["reference_wall_seconds"])),
                "mean_wall_speedup": float(np.mean(ng["reference_wall_seconds"] / np.maximum(ng["wall_seconds"], 1e-12))),
                "median_wall_speedup": float(np.median(ng["reference_wall_seconds"] / np.maximum(ng["wall_seconds"], 1e-12))),
            }
            per_n.append(record)
            summary_rows.append(record)
        atom_slope = bootstrap_log2_slope(xs, work_groups, rng, bootstrap_replicates)
        reference_slope = bootstrap_log2_slope(xs, reference_groups, rng, bootstrap_replicates)
        largest = max(per_n, key=lambda row: row["n"])
        key = f"{family}|p={p}|{algorithm}"
        detailed[key] = {
            "code_family": family,
            "p": float(p),
            "algorithm": algorithm,
            "per_n": per_n,
            "work_log2_slope": atom_slope[0],
            "work_slope_ci95": [atom_slope[1], atom_slope[2]],
            "reference_log2_slope": reference_slope[0],
            "reference_slope_ci95": [reference_slope[1], reference_slope[2]],
            "slope_advantage": reference_slope[0] - atom_slope[0],
            "largest_n": largest,
        }

    summary_frame = pd.DataFrame(summary_rows)
    summary_frame.to_csv(output_dir / "04_one_deletion_summary.csv", index=False)

    path_summary: dict[str, Any] = {}
    for (family, p), group in path_frame.groupby(["code_family", "p"]):
        path_summary[f"{family}|p={p}"] = {
            "trials": len(group),
            "pathwise_ml_disagreement_fraction": float(np.mean(~group["path_exact_ml"])),
            "median_histories": float(np.median(group["path_histories"])),
        }

    payload = {
        "representations": detailed,
        "pathwise": path_summary,
        "code_metadata": code_metadata,
        "exact_all": bool(frame["exact"].all()),
    }
    write_json(output_dir / "04_one_deletion_scaling.json", payload)
    return payload
