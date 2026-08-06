from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from .baselines import exhaustive_insertion_ml
from .channels import InsertionChannel, sample_insertion_channel
from .experiments_common import (
    effective_history_count,
    history_entropy_bits,
    insertion_history_components,
    make_code,
    result_row_base,
)
from .history_decoder import history_fiber_decode, insertion_history_model, pathwise_first_hit
from .utils import bootstrap_log2_slope, write_json


def run_one_insertion_scaling(
    output_dir: Path,
    rng: np.random.Generator,
    blocklengths: Sequence[int],
    target_rate: float,
    substitution_probabilities: Sequence[float],
    code_families: Sequence[str],
    trial_schedule: dict[str, int],
    max_histories: int,
    bootstrap_replicates: int,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    path_rows: list[dict[str, Any]] = []
    for family in code_families:
        for n in blocklengths:
            k = max(1, min(n - 1, int(round(target_rate * n))))
            code = make_code(family, n, k, rng, label="_I1")
            trials = int(trial_schedule.get(str(n), 20))
            for p in substitution_probabilities:
                channel = InsertionChannel(n=n, substitution_probability=float(p))
                for trial in range(trials):
                    true_message = int(rng.integers(0, code.size))
                    transmitted = int(code.codewords_int[true_message])
                    received, position, inserted, substitution_mask = sample_insertion_channel(
                        transmitted, channel, rng
                    )
                    reference = exhaustive_insertion_ml(code, received, channel)
                    result = history_fiber_decode(
                        insertion_history_model(received, channel),
                        code,
                        reference.tie_set,
                        reference.decision,
                        reference.work,
                        max_histories=max_histories,
                    )
                    path_decision, path_work = pathwise_first_hit(
                        insertion_history_model(received, channel),
                        code,
                        max_histories=max_histories,
                    )
                    components = insertion_history_components(transmitted, received, channel)
                    balanced = result.work.scalar("balanced")
                    pessimistic = result.work.scalar("pessimistic")
                    reference_balanced = reference.work.scalar("balanced")
                    reference_pessimistic = reference.work.scalar("pessimistic")
                    base = result_row_base(code, channel.label, n, trial, true_message, reference.tie_set)
                    rows.append(
                        {
                            **base,
                            "p": p,
                            "insert_position": position,
                            "inserted_bit": inserted,
                            "substitution_weight": int(substitution_mask).bit_count(),
                            "algorithm": "HISTORY_L0",
                            "decision": result.decision,
                            "exact": result.exact,
                            "certified": result.certified,
                            "fallback": result.fallback_used,
                            "work_balanced": balanced,
                            "work_pessimistic": pessimistic,
                            "reference_work_balanced": reference_balanced,
                            "reference_work_pessimistic": reference_pessimistic,
                            "reference_wall_seconds": reference.work.wall_seconds,
                            "speedup_balanced": reference_balanced / max(balanced, 1e-12),
                            "speedup_pessimistic": reference_pessimistic / max(pessimistic, 1e-12),
                            "candidate_score_speedup": code.size / max(1, result.work.exact_score_calls),
                            "membership_fraction_of_codebook": result.work.membership_queries / code.size,
                            "history_components": result.work.history_components,
                            "distinct_candidates": result.work.distinct_candidates,
                            "duplicate_histories": result.work.duplicate_histories,
                            "membership_queries": result.work.membership_queries,
                            "exact_score_calls": result.work.exact_score_calls,
                            "wall_seconds": result.work.wall_seconds,
                            "wall_speedup": reference.work.wall_seconds / max(result.work.wall_seconds, 1e-12),
                            "true_effective_history_count": effective_history_count(components),
                            "true_history_entropy_bits": history_entropy_bits(components),
                        }
                    )
                    path_rows.append(
                        {
                            **base,
                            "p": p,
                            "path_exact_ml": path_decision in reference.tie_set if path_decision is not None else False,
                            "path_histories": path_work.history_components,
                        }
                    )
    frame = pd.DataFrame(rows)
    path_frame = pd.DataFrame(path_rows)
    frame.to_csv(output_dir / "05_one_insertion_trials.csv.gz", index=False, compression="gzip")
    path_frame.to_csv(output_dir / "05_one_insertion_pathwise.csv.gz", index=False, compression="gzip")

    summary_rows = []
    details: dict[str, Any] = {}
    for (family, p), group in frame.groupby(["code_family", "p"]):
        xs = []
        work_groups = []
        reference_groups = []
        per_n = []
        for n, ng in group.groupby("n"):
            xs.append(float(n))
            work = ng["work_balanced"].to_numpy(float)
            reference = ng["reference_work_balanced"].to_numpy(float)
            work_groups.append(work)
            reference_groups.append(reference)
            record = {
                "code_family": family,
                "p": float(p),
                "n": int(n),
                "code_size": int(ng["code_size"].iloc[0]),
                "mean_work": float(np.mean(work)),
                "p95_work": float(np.percentile(work, 95)),
                "mean_reference_work": float(np.mean(reference)),
                "mean_speedup": float(np.mean(ng["speedup_balanced"])),
                "p05_speedup": float(np.percentile(ng["speedup_balanced"], 5)),
                "completion_fraction": float(np.mean(~ng["fallback"])),
                "exact_fraction": float(np.mean(ng["exact"])),
                "median_candidate_score_speedup": float(np.median(ng["candidate_score_speedup"])),
                "median_membership_fraction": float(np.median(ng["membership_fraction_of_codebook"])),
                "p99_membership_fraction": float(np.percentile(ng["membership_fraction_of_codebook"], 99)),
                "median_histories": float(np.median(ng["history_components"])),
                "p99_histories": float(np.percentile(ng["history_components"], 99)),
                "mean_wall_seconds": float(np.mean(ng["wall_seconds"])),
                "mean_reference_wall_seconds": float(np.mean(ng["reference_wall_seconds"])),
                "mean_wall_speedup": float(np.mean(ng["wall_speedup"])),
                "median_wall_speedup": float(np.median(ng["wall_speedup"])),
            }
            per_n.append(record)
            summary_rows.append(record)
        slope = bootstrap_log2_slope(xs, work_groups, rng, bootstrap_replicates)
        ref_slope = bootstrap_log2_slope(xs, reference_groups, rng, bootstrap_replicates)
        largest = max(per_n, key=lambda row: row["n"])
        details[f"{family}|p={p}"] = {
            "code_family": family,
            "p": float(p),
            "per_n": per_n,
            "work_log2_slope": slope[0],
            "work_slope_ci95": [slope[1], slope[2]],
            "reference_log2_slope": ref_slope[0],
            "reference_slope_ci95": [ref_slope[1], ref_slope[2]],
            "slope_advantage": ref_slope[0] - slope[0],
            "largest_n": largest,
        }
    pd.DataFrame(summary_rows).to_csv(output_dir / "05_one_insertion_summary.csv", index=False)
    path_summary = {
        f"{family}|p={p}": {
            "trials": len(group),
            "pathwise_ml_disagreement_fraction": float(np.mean(~group["path_exact_ml"])),
        }
        for (family, p), group in path_frame.groupby(["code_family", "p"])
    }
    payload = {"representations": details, "pathwise": path_summary, "exact_all": bool(frame["exact"].all())}
    write_json(output_dir / "05_one_insertion_scaling.json", payload)
    return payload
