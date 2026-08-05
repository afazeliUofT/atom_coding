from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .atlas import candidate_representations
from .contract import decision_contract
from .channels import (
    ChannelSpec,
    additive_qary_representation,
    asymmetric_erasure_stuck,
    bac,
    noncyclic_reversible_action_channel,
)
from .codes import BinaryLinearCode, QaryRandomCodebook
from .decoder import (
    binary_product_residual_decode,
    qary_reversible_first_hit_decode,
    strongest_binary_exact_reference,
)
from .metrics import (
    bootstrap_slope_ci,
    channel_capacity_blahut_arimoto,
    entropy_bits,
    representation_summary,
)
from .representations import bac_representations, unique_representations
from .simulation import sample_binary_linear_trial, sample_qary_trial
from .utils import write_json


def _select_binary_representations(
    spec: ChannelSpec,
    rng: np.random.Generator,
    count: int,
    vertex_count: int,
) -> list:
    if spec.family == "binary_asymmetric":
        reps = bac_representations(
            float(spec.metadata["a"]),
            float(spec.metadata["b"]),
            grid_points=15,
        )
    else:
        reps = candidate_representations(spec, rng, vertex_count=vertex_count)

    scored = []
    for rep in reps:
        summary = representation_summary(rep, spec.matrix)
        proxy = (
            math.log2(float(summary["kappa"]))
            + float(summary["transition_degeneracy_bits"])
            + 0.02 * rep.support_size
        )
        scored.append((proxy, rep, summary))
    selected = []
    # Best joint proxy, best kappa, best degeneracy, independent coupling, and endpoints for BAC.
    selected.append(min(scored, key=lambda item: item[0])[1])
    selected.append(min(scored, key=lambda item: float(item[2]["kappa"]))[1])
    selected.append(min(scored, key=lambda item: float(item[2]["transition_degeneracy_bits"]))[1])
    independent = [item[1] for item in scored if item[1].metadata.get("construction") == "independent_row_coupling"]
    selected.extend(independent[:1])
    endpoints = [
        item[1]
        for item in scored
        if item[1].metadata.get("is_lower_endpoint") or item[1].metadata.get("is_upper_endpoint")
    ]
    selected.extend(endpoints)
    deduped = unique_representations(selected)
    return deduped[:count]


def run_reversible_action_scaling(
    output_dir: Path,
    rng: np.random.Generator,
    blocklengths: list[int],
    normalized_rate: float,
    trials: int,
    max_atoms: int,
    bootstrap_replicates: int,
) -> dict[str, Any]:
    weights = [0.67, 0.22, 0.11]
    additive_spec, additive_rep = additive_qary_representation(5, weights)
    action_spec, action_rep = noncyclic_reversible_action_channel()
    channel_reps = [(additive_spec, additive_rep), (action_spec, action_rep)]
    rows: list[dict[str, Any]] = []

    for n in blocklengths:
        code_size = max(2, int(round(5 ** (normalized_rate * n))))
        code_size = min(code_size, 5**n)
        code = QaryRandomCodebook.random(5, n, code_size, rng, name=f"RANDOM_q5_n{n}_M{code_size}")
        for spec, rep in channel_reps:
            trial_data = [sample_qary_trial(spec.matrix, code, rng) for _ in range(trials)]
            for trial_index, (true_message, _, received) in enumerate(trial_data):
                result = qary_reversible_first_hit_decode(
                    spec.matrix,
                    rep,
                    code,
                    received,
                    max_atoms=max_atoms,
                )
                direct_work = result.notes["direct_work"]
                rows.append(
                    {
                        "channel": spec.name,
                        "family": spec.family,
                        "n": n,
                        "q": 5,
                        "code_size": code.size,
                        "normalized_rate": normalized_rate,
                        "rate_bits_per_symbol": math.log2(code.size) / n,
                        "trial": trial_index,
                        "true_message": true_message,
                        "decision": result.decision,
                        "ml_contains_true": true_message in result.ml_tie_set,
                        "exact": result.exact,
                        "certified": result.certified,
                        "fallback": result.fallback_used,
                        "atom_work_optimistic": result.work.scalar("optimistic"),
                        "atom_work_balanced": result.work.scalar("balanced"),
                        "atom_work_pessimistic": result.work.scalar("pessimistic"),
                        "direct_work_balanced": float(direct_work["scalar_balanced"]),
                        "speedup_balanced": float(direct_work["scalar_balanced"]) / max(result.work.scalar("balanced"), 1e-12),
                        "atoms_processed": result.work.atoms_processed,
                        "membership_queries": result.work.membership_queries,
                        "wall_seconds": result.work.wall_seconds,
                    }
                )
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "04_reversible_action_trials.csv.gz", index=False, compression="gzip")

    grouped_rows = []
    summaries: dict[str, Any] = {}
    for channel, group in frame.groupby("channel"):
        per_n = []
        work_groups = []
        direct_groups = []
        xs = []
        for n, ng in group.groupby("n"):
            xs.append(float(n))
            work_values = ng["atom_work_balanced"].to_numpy(float)
            direct_values = ng["direct_work_balanced"].to_numpy(float)
            work_groups.append(work_values)
            direct_groups.append(direct_values)
            record = {
                "channel": channel,
                "n": int(n),
                "mean_atom_work_balanced": float(np.mean(work_values)),
                "median_atom_work_balanced": float(np.median(work_values)),
                "p95_atom_work_balanced": float(np.percentile(work_values, 95)),
                "mean_direct_work_balanced": float(np.mean(direct_values)),
                "mean_speedup_balanced": float(np.mean(ng["speedup_balanced"])),
                "completion_fraction": float(np.mean(~ng["fallback"])),
                "exact_fraction": float(np.mean(ng["exact"])),
                "mean_queries": float(np.mean(ng["membership_queries"])),
            }
            per_n.append(record)
            grouped_rows.append(record)
        atom_slope = bootstrap_slope_ci(xs, work_groups, rng, replicates=bootstrap_replicates)
        direct_slope = bootstrap_slope_ci(xs, direct_groups, rng, replicates=bootstrap_replicates)
        largest = max(per_n, key=lambda row: row["n"])
        summaries[channel] = {
            "per_n": per_n,
            "atom_log2_work_slope": atom_slope[0],
            "atom_slope_ci95": [atom_slope[1], atom_slope[2]],
            "direct_log2_work_slope": direct_slope[0],
            "direct_slope_ci95": [direct_slope[1], direct_slope[2]],
            "slope_advantage": direct_slope[0] - atom_slope[0],
            "largest_n_speedup": largest["mean_speedup_balanced"],
            "largest_n_completion": largest["completion_fraction"],
            "exact_all": bool(group["exact"].all()),
        }
    pd.DataFrame(grouped_rows).to_csv(output_dir / "04_reversible_action_summary.csv", index=False)
    thresholds = decision_contract()["reversible_positive_control"]
    pass_flags = {
        channel: bool(
            summary["exact_all"]
            and summary["largest_n_completion"] >= float(thresholds["minimum_completion_fraction"])
            and summary["slope_advantage"] >= float(thresholds["minimum_log2_work_slope_advantage"])
            and summary["largest_n_speedup"] >= float(thresholds["minimum_largest_n_speedup"])
        )
        for channel, summary in summaries.items()
    }
    result_summary = {
        "channels": summaries,
        "pass_flags": pass_flags,
        "pass": all(pass_flags.values()),
        "interpretation": "positive control and possible reversible-action pivot only",
    }
    write_json(output_dir / "04_reversible_action_scaling.json", result_summary)
    return result_summary


def run_nonlatin_scaling(
    output_dir: Path,
    rng: np.random.Generator,
    blocklengths: list[int],
    rates: list[float],
    trials: int,
    max_atoms: int,
    representations_per_channel: int,
    vertex_count: int,
    bootstrap_replicates: int,
) -> dict[str, Any]:
    specs = [
        bac(0.12, 0.28, "BAC_MODERATE"),
        bac(0.25, 0.40, "BAC_HARD"),
        asymmetric_erasure_stuck(
            (0.75, 0.05, 0.20),
            (0.10, 0.65, 0.25),
            "AES_INJECTIVE_FRIENDLY",
        ),
        asymmetric_erasure_stuck(
            (0.40, 0.05, 0.55),
            (0.10, 0.25, 0.65),
            "AES_OVERLOADED_ERASURE",
        ),
    ]
    rows: list[dict[str, Any]] = []
    representation_catalog: dict[str, list[dict[str, Any]]] = {}

    for spec in specs:
        reps = _select_binary_representations(
            spec,
            rng,
            count=representations_per_channel,
            vertex_count=vertex_count,
        )
        representation_catalog[spec.name] = [
            {**rep.to_dict(), "summary": representation_summary(rep, spec.matrix)} for rep in reps
        ]
        for rate in rates:
            for n in blocklengths:
                k = max(1, min(n - 1, int(round(rate * n))))
                code = BinaryLinearCode.random_systematic(
                    n,
                    k,
                    rng,
                    name=f"RLC_{spec.name}_n{n}_k{k}",
                )
                trial_data = [sample_binary_linear_trial(spec.matrix, code, rng) for _ in range(trials)]
                trial_references = [
                    strongest_binary_exact_reference(spec.matrix, code, received)
                    for _, _, received in trial_data
                ]
                for rep in reps:
                    rep_summary = representation_summary(rep, spec.matrix)
                    for trial_index, (true_message, _, received) in enumerate(trial_data):
                        result = binary_product_residual_decode(
                            spec.matrix,
                            rep,
                            code,
                            received,
                            max_atoms=max_atoms,
                            reference=trial_references[trial_index],
                        )
                        direct_work = result.notes["direct_work"]
                        trellis_work = result.notes["trellis_work"]
                        reference_work = result.notes["reference_work"]
                        rows.append(
                            {
                                "channel": spec.name,
                                "family": spec.family,
                                "representation": rep.name,
                                "construction": rep.metadata.get("construction", "unspecified"),
                                "n": n,
                                "k": k,
                                "rate": k / n,
                                "target_rate": rate,
                                "trial": trial_index,
                                "true_message": true_message,
                                "decision": result.decision,
                                "ml_contains_true": true_message in result.ml_tie_set,
                                "exact": result.exact,
                                "certified": result.certified,
                                "fallback": result.fallback_used,
                                "atom_work_optimistic": result.work.scalar("optimistic"),
                                "atom_work_balanced": result.work.scalar("balanced"),
                                "atom_work_pessimistic": result.work.scalar("pessimistic"),
                                "direct_work_balanced": float(direct_work["scalar_balanced"]),
                                "trellis_work_balanced": float(trellis_work["scalar_balanced"]),
                                "reference_name": result.notes["reference_name"],
                                "reference_work_balanced": float(reference_work["scalar_balanced"]),
                                "speedup_balanced": float(reference_work["scalar_balanced"]) / max(result.work.scalar("balanced"), 1e-12),
                                "atoms_processed": result.work.atoms_processed,
                                "fiber_queries": result.work.fiber_queries,
                                "fiber_entries": result.work.fiber_entries,
                                "score_updates": result.work.score_updates,
                                "transition_degeneracy_bits": rep_summary["transition_degeneracy_bits"],
                                "kappa": rep_summary["kappa"],
                                "fiber_ceiling": rep_summary["random_code_fiber_ceiling_bits_per_symbol"],
                                "support_size": rep.support_size,
                                "residual_mass": result.residual_mass,
                                "wall_seconds": result.work.wall_seconds,
                            }
                        )
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "05_nonlatin_trials.csv.gz", index=False, compression="gzip")
    write_json(output_dir / "05_nonlatin_representation_catalog.json", representation_catalog)

    aggregate_rows = []
    rep_summaries: dict[str, Any] = {}
    for (channel, target_rate, representation), group in frame.groupby(
        ["channel", "target_rate", "representation"]
    ):
        xs: list[float] = []
        work_groups: list[np.ndarray] = []
        reference_groups: list[np.ndarray] = []
        pessimistic_groups: list[np.ndarray] = []
        per_n = []
        for n, ng in group.groupby("n"):
            xs.append(float(n))
            work = ng["atom_work_balanced"].to_numpy(float)
            pessimistic = ng["atom_work_pessimistic"].to_numpy(float)
            reference = ng["reference_work_balanced"].to_numpy(float)
            direct = ng["direct_work_balanced"].to_numpy(float)
            trellis = ng["trellis_work_balanced"].to_numpy(float)
            work_groups.append(work)
            pessimistic_groups.append(pessimistic)
            reference_groups.append(reference)
            record = {
                "channel": channel,
                "target_rate": float(target_rate),
                "representation": representation,
                "n": int(n),
                "actual_rate": float(ng["rate"].iloc[0]),
                "mean_atom_work_balanced": float(np.mean(work)),
                "mean_atom_work_pessimistic": float(np.mean(pessimistic)),
                "median_atom_work_balanced": float(np.median(work)),
                "p95_atom_work_balanced": float(np.percentile(work, 95)),
                "p99_atom_work_balanced": float(np.percentile(work, 99)),
                "mean_reference_work": float(np.mean(reference)),
                "mean_direct_work": float(np.mean(direct)),
                "mean_trellis_work": float(np.mean(trellis)),
                "dominant_reference": str(ng["reference_name"].mode().iloc[0]),
                "mean_speedup": float(np.mean(ng["speedup_balanced"])),
                "completion_fraction": float(np.mean(~ng["fallback"])),
                "exact_fraction": float(np.mean(ng["exact"])),
                "mean_atoms": float(np.mean(ng["atoms_processed"])),
                "mean_fiber_entries": float(np.mean(ng["fiber_entries"])),
                "kappa": float(ng["kappa"].iloc[0]),
                "transition_degeneracy_bits": float(ng["transition_degeneracy_bits"].iloc[0]),
                "fiber_ceiling": float(ng["fiber_ceiling"].iloc[0]),
            }
            per_n.append(record)
            aggregate_rows.append(record)
        atom_slope = bootstrap_slope_ci(xs, work_groups, rng, replicates=bootstrap_replicates)
        pessimistic_slope = bootstrap_slope_ci(xs, pessimistic_groups, rng, replicates=bootstrap_replicates)
        reference_slope = bootstrap_slope_ci(xs, reference_groups, rng, replicates=bootstrap_replicates)
        largest = max(per_n, key=lambda row: row["n"])
        key = f"{channel}|R={target_rate}|{representation}"
        rep_summaries[key] = {
            "channel": channel,
            "target_rate": float(target_rate),
            "representation": representation,
            "per_n": per_n,
            "atom_slope": atom_slope[0],
            "atom_slope_ci95": [atom_slope[1], atom_slope[2]],
            "pessimistic_slope": pessimistic_slope[0],
            "reference_slope": reference_slope[0],
            "slope_advantage": reference_slope[0] - atom_slope[0],
            "pessimistic_slope_advantage": reference_slope[0] - pessimistic_slope[0],
            "largest_n_speedup": largest["mean_speedup"],
            "largest_n_completion": largest["completion_fraction"],
            "exact_all": bool(group["exact"].all()),
            "kappa": largest["kappa"],
            "degeneracy": largest["transition_degeneracy_bits"],
            "fiber_ceiling": largest["fiber_ceiling"],
        }
    pd.DataFrame(aggregate_rows).to_csv(output_dir / "05_nonlatin_summary.csv", index=False)

    thresholds = decision_contract()["nonlatin_H3"]
    channel_pass: dict[str, bool] = {}
    channel_best: dict[str, Any] = {}
    for channel in frame["channel"].unique():
        candidates = [summary for summary in rep_summaries.values() if summary["channel"] == channel]
        passing = [
            summary
            for summary in candidates
            if summary["exact_all"]
            and summary["largest_n_completion"] >= float(thresholds["minimum_completion_fraction"])
            and summary["slope_advantage"] >= float(thresholds["minimum_log2_work_slope_advantage"])
            and summary["largest_n_speedup"] >= float(thresholds["minimum_largest_n_speedup"])
            and summary["pessimistic_slope_advantage"] >= float(thresholds["minimum_pessimistic_slope_advantage"])
        ]
        channel_pass[channel] = len(passing) > 0
        rank_pool = passing if passing else candidates
        best = max(
            rank_pool,
            key=lambda item: (
                item["slope_advantage"],
                item["largest_n_speedup"],
                item["largest_n_completion"],
            ),
        )
        channel_best[channel] = best

    summary = {
        "representations": rep_summaries,
        "channel_pass": channel_pass,
        "channel_best": channel_best,
        "passing_channels": int(sum(channel_pass.values())),
        "exact_all": bool(frame["exact"].all()),
    }
    write_json(output_dir / "05_nonlatin_scaling.json", summary)
    return summary
