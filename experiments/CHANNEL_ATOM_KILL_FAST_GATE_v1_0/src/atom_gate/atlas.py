from __future__ import annotations

import itertools
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .channels import ChannelSpec, default_channel_suite
from .contract import decision_contract
from .decoder import one_shot_residual_decode
from .metrics import output_distribution_uniform, representation_summary
from .models import Representation, WorkVector
from .representations import (
    bac_representations,
    bec_representations,
    common_uniform_candidates,
    independent_row_coupling,
    max_injective_representation,
    min_ambiguity_representation,
    min_kappa_representation,
    sample_vertex_representations,
    unique_representations,
)
from .utils import write_json


def candidate_representations(
    channel_spec: ChannelSpec,
    rng: np.random.Generator,
    vertex_count: int,
) -> list[Representation]:
    w = channel_spec.matrix
    reps: list[Representation] = [independent_row_coupling(w)]
    reps.extend(common_uniform_candidates(w, max_candidates=12))
    try:
        reps.append(min_kappa_representation(w))
        reps.append(min_ambiguity_representation(w))
        reps.append(max_injective_representation(w))
    except Exception:
        # Some degenerate channels can make redundant LPs numerically awkward; vertices below still test them.
        pass
    if channel_spec.family == "binary_asymmetric":
        a = float(channel_spec.metadata["a"])
        b = float(channel_spec.metadata["b"])
        reps.extend(bac_representations(a, b, grid_points=11))
    if channel_spec.family == "additive_control" and w.shape == (2, 2):
        a = float(w[0, 1])
        b = float(w[1, 0])
        reps.extend(bac_representations(a, b, grid_points=7))
    if channel_spec.family == "erasure_control":
        reps.extend(bec_representations(float(channel_spec.metadata["epsilon"]), grid_points=9))
    reps.extend(sample_vertex_representations(w, rng, count=vertex_count, prefix=f"{channel_spec.name}_V"))
    result = unique_representations(reps)
    for rep in result:
        rep.verify(w)
    return result


def order_candidates(rep: Representation, rng: np.random.Generator) -> list[tuple[int, ...]]:
    s = rep.support_size
    mass = tuple(sorted(range(s), key=lambda i: (-float(rep.weights[i]), rep.maps[i].outputs)))
    candidates = [mass, tuple(reversed(mass))]
    if s <= 6:
        candidates.extend(itertools.permutations(range(s)))
    else:
        for _ in range(32):
            candidates.append(tuple(int(v) for v in rng.permutation(s)))
    unique: list[tuple[int, ...]] = []
    seen: set[tuple[int, ...]] = set()
    for order in candidates:
        order = tuple(order)
        if order not in seen:
            seen.add(order)
            unique.append(order)
    return unique


def expected_one_shot_work(
    channel: np.ndarray,
    rep: Representation,
    rng: np.random.Generator,
) -> dict[str, object]:
    py = output_distribution_uniform(channel)
    best: dict[str, tuple[float, tuple[int, ...], dict[str, float]]] = {}
    exact_all = True
    for order in order_candidates(rep, rng):
        weighted = {model: 0.0 for model in ("optimistic", "balanced", "pessimistic")}
        atoms = 0.0
        fibers = 0.0
        residual = 0.0
        certified_prob = 0.0
        for y, probability in enumerate(py):
            if probability <= 0.0:
                continue
            result = one_shot_residual_decode(channel, rep, y, order)
            exact_all &= result.exact
            for model in weighted:
                weighted[model] += float(probability) * result.work.scalar(model)
            atoms += float(probability) * result.work.atoms_processed
            fibers += float(probability) * result.work.fiber_entries
            residual += float(probability) * result.residual_mass
            certified_prob += float(probability) * float(result.certified)
        for model in weighted:
            current = best.get(model)
            stats = {
                "atoms": atoms,
                "fiber_entries": fibers,
                "residual": residual,
                "certified_probability": certified_prob,
            }
            if current is None or weighted[model] < current[0]:
                best[model] = (weighted[model], order, stats)
    return {
        "exact_all": exact_all,
        "best_work_optimistic": best["optimistic"][0],
        "best_work_balanced": best["balanced"][0],
        "best_work_pessimistic": best["pessimistic"][0],
        "best_order_optimistic": list(best["optimistic"][1]),
        "best_order_balanced": list(best["balanced"][1]),
        "best_order_pessimistic": list(best["pessimistic"][1]),
        "best_balanced_atoms": best["balanced"][2]["atoms"],
        "best_balanced_fiber_entries": best["balanced"][2]["fiber_entries"],
        "best_balanced_certified_probability": best["balanced"][2]["certified_probability"],
    }


def run_atlas(
    output_dir: Path,
    rng: np.random.Generator,
    random_channels: int,
    vertex_count: int,
) -> dict[str, object]:
    channels = default_channel_suite(rng, random_count=random_channels)
    rows: list[dict[str, object]] = []
    rep_dump: dict[str, list[dict[str, object]]] = {}
    for spec in channels:
        reps = candidate_representations(spec, rng, vertex_count)
        rep_dump[spec.name] = [rep.to_dict() for rep in reps]
        for rep in reps:
            summary = representation_summary(rep, spec.matrix)
            one_shot = expected_one_shot_work(spec.matrix, rep, rng)
            rows.append(
                {
                    "channel": spec.name,
                    "family": spec.family,
                    "nonadditive": spec.nonadditive,
                    **summary,
                    **one_shot,
                    "construction": rep.metadata.get("construction", "unspecified"),
                    "is_natural_independent": rep.metadata.get("construction") == "independent_row_coupling",
                }
            )
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "02_small_dmc_atlas.csv", index=False)
    write_json(output_dir / "02_small_dmc_representations.json", rep_dump)

    thresholds = decision_contract()["atlas_H2"]
    channel_summaries = []
    witness_count = 0
    natural_improvement_count = 0
    for channel, group in frame.groupby("channel"):
        best = float(group["best_work_balanced"].min())
        worst = float(group["best_work_balanced"].max())
        natural_rows = group[group["is_natural_independent"]]
        natural = float(natural_rows["best_work_balanced"].iloc[0]) if not natural_rows.empty else float("nan")
        nonadditive = bool(group["nonadditive"].iloc[0])
        variation_ratio = worst / best if best > 0 else float("inf")
        natural_ratio = natural / best if np.isfinite(natural) and best > 0 else float("nan")
        eligible_nonadditive = nonadditive and str(group["family"].iloc[0]) != "negative_control"
        if eligible_nonadditive and variation_ratio >= float(thresholds["minimum_worst_to_best_work_ratio"]):
            witness_count += 1
        if (
            eligible_nonadditive
            and np.isfinite(natural_ratio)
            and natural_ratio >= float(thresholds["minimum_independent_to_best_work_ratio"])
        ):
            natural_improvement_count += 1
        channel_summaries.append(
            {
                "channel": channel,
                "family": group["family"].iloc[0],
                "nonadditive": nonadditive,
                "representations": int(len(group)),
                "best_balanced_work": best,
                "worst_balanced_work": worst,
                "variation_ratio": variation_ratio,
                "independent_balanced_work": natural,
                "independent_to_best_ratio": natural_ratio,
                "best_representation": str(group.loc[group["best_work_balanced"].idxmin(), "representation"]),
            }
        )
    summary = {
        "exact_all": bool(frame["exact_all"].all()),
        "channels": len(channels),
        "representations": len(frame),
        "nonadditive_variation_witnesses": witness_count,
        "nonadditive_natural_improvement_witnesses": natural_improvement_count,
        "channel_summaries": channel_summaries,
    }
    write_json(output_dir / "02_small_dmc_atlas_summary.json", summary)
    pd.DataFrame(channel_summaries).to_csv(output_dir / "02_small_dmc_channel_summary.csv", index=False)
    return summary
