from __future__ import annotations

import itertools
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

from .atlas import candidate_representations
from .channels import asymmetric_erasure_stuck, bac
from .contract import decision_contract
from .codes import digits_to_int, int_to_digits
from .metrics import channel_capacity_blahut_arimoto, mutual_information, representation_summary
from .models import Representation
from .representations import bac_representations, unique_representations
from .simulation import top_product_atoms
from .utils import write_json


def confusability_edges(
    representation: Representation,
    blocklength: int,
    input_size: int,
    output_size: int,
    cumulative_target: float,
    max_atoms: int,
) -> tuple[set[tuple[int, int]], float, int]:
    ordered_rep, atoms, cumulative = top_product_atoms(
        representation,
        blocklength,
        cumulative_target=cumulative_target,
        max_atoms=max_atoms,
    )
    ambient = input_size**blocklength
    output_to_inputs: dict[int, list[int]] = {}
    for x_word in range(ambient):
        x_digits = int_to_digits(x_word, blocklength, input_size)
        images: set[int] = set()
        for atom_indices in atoms:
            y_digits = [
                ordered_rep.maps[atom_index](x_digits[coordinate])
                for coordinate, atom_index in enumerate(atom_indices)
            ]
            images.add(digits_to_int(y_digits, output_size))
        for y_word in images:
            output_to_inputs.setdefault(y_word, []).append(x_word)
    edges: set[tuple[int, int]] = set()
    for inputs in output_to_inputs.values():
        if len(inputs) < 2:
            continue
        for u, v in itertools.combinations(sorted(set(inputs)), 2):
            edges.add((u, v))
    return edges, cumulative, len(atoms)


def greedy_independent_set_size(vertex_count: int, edges: set[tuple[int, int]]) -> int:
    adjacency = [set() for _ in range(vertex_count)]
    for u, v in edges:
        adjacency[u].add(v)
        adjacency[v].add(u)
    remaining = set(range(vertex_count))
    independent: list[int] = []
    while remaining:
        vertex = min(remaining, key=lambda v: len(adjacency[v] & remaining))
        independent.append(vertex)
        remaining.remove(vertex)
        remaining.difference_update(adjacency[vertex])
    return len(independent)


def maximum_independent_set_milp(
    vertex_count: int,
    edges: set[tuple[int, int]],
    time_limit_seconds: float = 90.0,
) -> dict[str, Any]:
    if not edges:
        return {
            "size": vertex_count,
            "lower_bound": vertex_count,
            "optimal": True,
            "status": "edgeless",
            "mip_gap": 0.0,
        }
    edge_list = sorted(edges)
    row_indices = []
    col_indices = []
    values = []
    for row, (u, v) in enumerate(edge_list):
        row_indices.extend([row, row])
        col_indices.extend([u, v])
        values.extend([1.0, 1.0])
    matrix = coo_matrix(
        (values, (row_indices, col_indices)),
        shape=(len(edge_list), vertex_count),
    ).tocsr()
    constraint = LinearConstraint(matrix, -np.inf, np.ones(len(edge_list)))
    result = milp(
        c=-np.ones(vertex_count),
        integrality=np.ones(vertex_count),
        bounds=Bounds(np.zeros(vertex_count), np.ones(vertex_count)),
        constraints=constraint,
        options={"time_limit": time_limit_seconds, "mip_rel_gap": 0.0, "presolve": True},
    )
    greedy = greedy_independent_set_size(vertex_count, edges)
    if result.x is not None:
        found = int(round(float(np.sum(result.x > 0.5))))
        lower = max(greedy, found)
    else:
        lower = greedy
    optimal = bool(result.success and getattr(result, "mip_gap", 1.0) <= 1e-9)
    size = lower if not optimal else int(round(-float(result.fun)))
    return {
        "size": size,
        "lower_bound": lower,
        "optimal": optimal,
        "status": str(result.message),
        "mip_gap": float(getattr(result, "mip_gap", float("nan"))),
    }


def _rate_representations(spec, rng, count: int, vertex_count: int) -> list[Representation]:
    if spec.family == "binary_asymmetric":
        reps = bac_representations(float(spec.metadata["a"]), float(spec.metadata["b"]), grid_points=17)
    else:
        reps = candidate_representations(spec, rng, vertex_count=vertex_count)
    summaries = [(representation_summary(rep, spec.matrix), rep) for rep in reps]
    selected = [min(summaries, key=lambda item: float(item[0]["kappa"]))[1]]
    independent = [rep for _, rep in summaries if rep.metadata.get("construction") == "independent_row_coupling"]
    selected.extend(independent[:1])
    selected.append(min(summaries, key=lambda item: float(item[0]["transition_degeneracy_bits"]))[1])
    return unique_representations(selected)[:count]


def run_rate_survival(
    output_dir: Path,
    rng: np.random.Generator,
    blocklengths: list[int],
    cumulative_targets: list[float],
    max_atoms: int,
    reps_per_channel: int,
    vertex_count: int,
    milp_time_limit: float,
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
    analytic_rows: list[dict[str, Any]] = []
    for spec in specs:
        capacity, px, _ = channel_capacity_blahut_arimoto(spec.matrix)
        uniform_input = np.full(spec.matrix.shape[0], 1.0 / spec.matrix.shape[0])
        uniform_information = mutual_information(spec.matrix, uniform_input)
        reps = _rate_representations(spec, rng, reps_per_channel, vertex_count)
        for rep in reps:
            summary = representation_summary(rep, spec.matrix)
            analytic_rows.append(
                {
                    "channel": spec.name,
                    "family": spec.family,
                    "representation": rep.name,
                    "construction": rep.metadata.get("construction", "unspecified"),
                    "capacity": capacity,
                    "uniform_input_mutual_information": uniform_information,
                    "capacity_input_distribution": px.tolist(),
                    "kappa": summary["kappa"],
                    "fiber_ceiling": summary["random_code_fiber_ceiling_bits_per_symbol"],
                    "fiber_ceiling_to_capacity": float(summary["random_code_fiber_ceiling_bits_per_symbol"]) / max(capacity, 1e-12),
                    "fiber_ceiling_to_uniform_information": float(summary["random_code_fiber_ceiling_bits_per_symbol"]) / max(uniform_information, 1e-12),
                    "degeneracy": summary["transition_degeneracy_bits"],
                    "injective_mass": summary["injective_mass"],
                }
            )
            for n in blocklengths:
                vertex_count_graph = 2**n
                for target in cumulative_targets:
                    edges, mass, atom_count = confusability_edges(
                        rep,
                        blocklength=n,
                        input_size=2,
                        output_size=spec.matrix.shape[1],
                        cumulative_target=target,
                        max_atoms=max_atoms,
                    )
                    solution = maximum_independent_set_milp(
                        vertex_count_graph,
                        edges,
                        time_limit_seconds=milp_time_limit,
                    )
                    achieved_rate = math.log2(max(1, solution["lower_bound"])) / n
                    rows.append(
                        {
                            "channel": spec.name,
                            "family": spec.family,
                            "representation": rep.name,
                            "construction": rep.metadata.get("construction", "unspecified"),
                            "n": n,
                            "target_cumulative_mass": target,
                            "actual_cumulative_mass": mass,
                            "atoms_in_likely_set": atom_count,
                            "vertices": vertex_count_graph,
                            "edges": len(edges),
                            "independent_set_size": solution["size"],
                            "independent_set_lower_bound": solution["lower_bound"],
                            "milp_optimal": solution["optimal"],
                            "mip_gap": solution["mip_gap"],
                            "atom_separating_rate": achieved_rate,
                            "capacity": capacity,
                            "uniform_input_mutual_information": uniform_information,
                            "rate_to_capacity": achieved_rate / max(capacity, 1e-12),
                            "rate_to_uniform_information": achieved_rate / max(uniform_information, 1e-12),
                            "fiber_ceiling": summary["random_code_fiber_ceiling_bits_per_symbol"],
                            "fiber_ceiling_to_capacity": float(summary["random_code_fiber_ceiling_bits_per_symbol"]) / max(capacity, 1e-12),
                            "fiber_ceiling_to_uniform_information": float(summary["random_code_fiber_ceiling_bits_per_symbol"]) / max(uniform_information, 1e-12),
                        }
                    )
    analytic = pd.DataFrame(analytic_rows)
    graph = pd.DataFrame(rows)
    analytic.to_csv(output_dir / "06_rate_survival_analytic.csv", index=False)
    graph.to_csv(output_dir / "06_rate_survival_graph.csv", index=False)

    thresholds = decision_contract()["rate_survival_H4"]
    channel_pass: dict[str, bool] = {}
    channel_best: dict[str, Any] = {}
    for channel in analytic["channel"].unique():
        a_group = analytic[analytic["channel"] == channel]
        candidates: list[dict[str, Any]] = []
        for _, analytic_row in a_group.iterrows():
            rep_name = str(analytic_row["representation"])
            g_group = graph[
                (graph["channel"] == channel)
                & (graph["representation"] == rep_name)
            ]
            if g_group.empty:
                continue
            largest_n = int(g_group["n"].max())
            largest_target = float(g_group["target_cumulative_mass"].max())
            final = g_group[
                (g_group["n"] == largest_n)
                & (g_group["target_cumulative_mass"] == largest_target)
            ].iloc[0]
            uniform_information = float(analytic_row["uniform_input_mutual_information"])
            fiber_ratio = float(analytic_row["fiber_ceiling_to_uniform_information"])
            rate_ratio = float(final["rate_to_uniform_information"])
            required_mass = min(
                float(thresholds["maximum_required_actual_likely_atom_mass"]),
                largest_target - 1e-9,
            )
            mass_ok = float(final["actual_cumulative_mass"]) >= required_mass
            joint_score = min(1.0, fiber_ratio, rate_ratio) if mass_ok else -1.0
            candidates.append(
                {
                    "best_representation": rep_name,
                    "capacity": float(analytic_row["capacity"]),
                    "uniform_input_mutual_information": uniform_information,
                    "fiber_ceiling": float(analytic_row["fiber_ceiling"]),
                    "fiber_ceiling_to_capacity": float(analytic_row["fiber_ceiling_to_capacity"]),
                    "fiber_ceiling_to_uniform_information": fiber_ratio,
                    "largest_n": largest_n,
                    "target_mass": largest_target,
                    "actual_mass": float(final["actual_cumulative_mass"]),
                    "atom_separating_rate_lower_bound": float(final["atom_separating_rate"]),
                    "rate_to_capacity": float(final["rate_to_capacity"]),
                    "rate_to_uniform_information": rate_ratio,
                    "milp_optimal": bool(final["milp_optimal"]),
                    "joint_score": joint_score,
                    "mass_ok": mass_ok,
                }
            )
        if not candidates:
            channel_pass[channel] = False
            channel_best[channel] = {"pass": False, "reason": "no evaluable representations"}
            continue
        best = max(
            candidates,
            key=lambda item: (
                item["joint_score"],
                min(1.0, item["rate_to_uniform_information"]),
                min(1.0, item["fiber_ceiling_to_uniform_information"]),
                item["actual_mass"],
            ),
        )
        passed = bool(
            best["fiber_ceiling_to_uniform_information"]
            >= float(thresholds["minimum_fiber_ceiling_fraction_of_uniform_information"])
            and best["rate_to_uniform_information"]
            >= float(thresholds["minimum_likely_atom_code_rate_fraction_of_uniform_information"])
            and best["mass_ok"]
        )
        best["pass"] = passed
        channel_pass[channel] = passed
        channel_best[channel] = best
    summary = {
        "channel_pass": channel_pass,
        "channel_best": channel_best,
        "passing_channels": int(sum(channel_pass.values())),
    }
    write_json(output_dir / "06_rate_survival.json", summary)
    return summary
