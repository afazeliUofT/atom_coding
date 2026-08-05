from __future__ import annotations

import itertools
from collections import OrderedDict
from typing import Iterable, Sequence

import numpy as np
from scipy.optimize import linprog

from .models import DeterministicMap, Representation


def enumerate_maps(input_size: int, output_size: int) -> list[DeterministicMap]:
    return [DeterministicMap(tuple(outputs)) for outputs in itertools.product(range(output_size), repeat=input_size)]


def map_channel_column(atom: DeterministicMap, output_size: int) -> np.ndarray:
    out = np.zeros((atom.input_size, output_size), dtype=float)
    for x, y in enumerate(atom.outputs):
        out[x, y] = 1.0
    return out


def decomposition_constraint_matrix(
    maps: Sequence[DeterministicMap],
    input_size: int,
    output_size: int,
) -> np.ndarray:
    columns = [map_channel_column(atom, output_size).reshape(-1) for atom in maps]
    return np.column_stack(columns)


def representation_from_lambda(
    name: str,
    maps: Sequence[DeterministicMap],
    weights: Sequence[float],
    metadata: dict[str, object] | None = None,
    atol: float = 1e-10,
) -> Representation:
    chosen_maps: list[DeterministicMap] = []
    chosen_weights: list[float] = []
    for atom, weight in zip(maps, weights, strict=True):
        if float(weight) > atol:
            chosen_maps.append(atom)
            chosen_weights.append(float(weight))
    if not chosen_maps:
        raise ValueError("LP returned no positive atoms")
    arr = np.asarray(chosen_weights, dtype=float)
    arr /= arr.sum()
    return Representation(name, chosen_maps, arr, metadata or {}).reduced()


def solve_representation_lp(
    channel: np.ndarray,
    objective: Sequence[float],
    name: str,
    maps: Sequence[DeterministicMap] | None = None,
    maximize: bool = False,
    metadata: dict[str, object] | None = None,
) -> Representation:
    channel = np.asarray(channel, dtype=float)
    m, qy = channel.shape
    all_maps = list(maps) if maps is not None else enumerate_maps(m, qy)
    a_eq = decomposition_constraint_matrix(all_maps, m, qy)
    b_eq = channel.reshape(-1)
    c = np.asarray(objective, dtype=float)
    if c.shape != (len(all_maps),):
        raise ValueError("Objective length does not match number of maps")
    if maximize:
        c = -c
    result = linprog(
        c,
        A_eq=a_eq,
        b_eq=b_eq,
        bounds=[(0.0, None)] * len(all_maps),
        method="highs",
    )
    if not result.success:
        raise RuntimeError(f"Representation LP failed: {result.message}")
    rep = representation_from_lambda(name, all_maps, result.x, metadata=metadata)
    rep.verify(channel)
    return rep


def independent_row_coupling(channel: np.ndarray, name: str = "INDEPENDENT_ROWS") -> Representation:
    channel = np.asarray(channel, dtype=float)
    m, qy = channel.shape
    maps = enumerate_maps(m, qy)
    weights = []
    for atom in maps:
        weight = 1.0
        for x, y in enumerate(atom.outputs):
            weight *= float(channel[x, y])
        weights.append(weight)
    rep = representation_from_lambda(name, maps, weights, {"construction": "independent_row_coupling"})
    rep.verify(channel)
    return rep


def common_uniform_coupling(
    channel: np.ndarray,
    row_orders: Sequence[Sequence[int]] | None = None,
    name: str = "COMMON_UNIFORM",
) -> Representation:
    """Couple rows using one U~Uniform[0,1] and declared output orders."""
    channel = np.asarray(channel, dtype=float)
    m, qy = channel.shape
    orders = [tuple(range(qy)) for _ in range(m)] if row_orders is None else [tuple(v) for v in row_orders]
    if len(orders) != m or any(sorted(order) != list(range(qy)) for order in orders):
        raise ValueError("Each row order must be a permutation of output labels")

    breakpoints = {0.0, 1.0}
    cumulative_rows: list[list[tuple[float, int]]] = []
    for x in range(m):
        cumulative = 0.0
        row: list[tuple[float, int]] = []
        for y in orders[x]:
            cumulative += float(channel[x, y])
            row.append((cumulative, y))
            breakpoints.add(round(cumulative, 15))
        cumulative_rows.append(row)
    points = sorted(breakpoints)
    maps: list[DeterministicMap] = []
    weights: list[float] = []
    for left, right in zip(points[:-1], points[1:], strict=True):
        if right - left <= 1e-14:
            continue
        u = (left + right) / 2.0
        outputs: list[int] = []
        for row in cumulative_rows:
            for threshold, y in row:
                if u <= threshold + 1e-13:
                    outputs.append(y)
                    break
        maps.append(DeterministicMap(tuple(outputs)))
        weights.append(right - left)
    rep = Representation(name, maps, np.asarray(weights), {"construction": "common_uniform", "orders": [list(v) for v in orders]}).reduced()
    rep.verify(channel)
    return rep


def common_uniform_candidates(channel: np.ndarray, max_candidates: int = 24) -> list[Representation]:
    channel = np.asarray(channel, dtype=float)
    m, qy = channel.shape
    permutations = list(itertools.permutations(range(qy)))
    candidates: list[Representation] = []
    seen: set[tuple[tuple[tuple[int, ...], float], ...]] = set()
    for index, orders in enumerate(itertools.product(permutations, repeat=m)):
        if index >= max_candidates:
            break
        rep = common_uniform_coupling(channel, orders, name=f"COMMON_UNIFORM_{index:03d}")
        key = tuple(sorted((atom.outputs, round(float(weight), 12)) for atom, weight in zip(rep.maps, rep.weights, strict=True)))
        if key not in seen:
            candidates.append(rep)
            seen.add(key)
    return candidates


def map_kappa(atom: DeterministicMap, output_size: int | None = None) -> float:
    qy = atom.output_size if output_size is None else output_size
    counts = np.zeros(qy, dtype=int)
    for y in atom.outputs:
        counts[y] += 1
    return float(np.sum(counts.astype(float) ** 2) / atom.input_size)


def map_uniform_fiber_ambiguity(atom: DeterministicMap, output_size: int | None = None) -> float:
    qy = atom.output_size if output_size is None else output_size
    counts = np.zeros(qy, dtype=int)
    for y in atom.outputs:
        counts[y] += 1
    value = 0.0
    for count in counts:
        if count > 0:
            value += (count / atom.input_size) * np.log2(count)
    return float(value)


def min_kappa_representation(channel: np.ndarray, name: str = "LP_MIN_KAPPA") -> Representation:
    channel = np.asarray(channel, dtype=float)
    maps = enumerate_maps(*channel.shape)
    objective = [map_kappa(atom, channel.shape[1]) for atom in maps]
    return solve_representation_lp(
        channel,
        objective,
        name,
        maps=maps,
        metadata={"construction": "linear_program", "objective": "min_kappa"},
    )


def min_ambiguity_representation(channel: np.ndarray, name: str = "LP_MIN_AMBIGUITY") -> Representation:
    channel = np.asarray(channel, dtype=float)
    maps = enumerate_maps(*channel.shape)
    objective = [map_uniform_fiber_ambiguity(atom, channel.shape[1]) for atom in maps]
    return solve_representation_lp(
        channel,
        objective,
        name,
        maps=maps,
        metadata={"construction": "linear_program", "objective": "min_uniform_fiber_ambiguity"},
    )


def max_injective_representation(channel: np.ndarray, name: str = "LP_MAX_INJECTIVE") -> Representation:
    channel = np.asarray(channel, dtype=float)
    maps = enumerate_maps(*channel.shape)
    objective = [1.0 if atom.is_injective() else 0.0 for atom in maps]
    return solve_representation_lp(
        channel,
        objective,
        name,
        maps=maps,
        maximize=True,
        metadata={"construction": "linear_program", "objective": "max_injective_mass"},
    )


def compact_max_injective_mass(channel: np.ndarray) -> float:
    """Compact LP from the proposal appendix; returns eta only."""
    channel = np.asarray(channel, dtype=float)
    m, qy = channel.shape
    # Variables are B_{m,y} followed by eta.
    n_b = m * qy
    n_var = n_b + 1
    c = np.zeros(n_var)
    c[-1] = -1.0

    a_eq = []
    b_eq = []
    for x in range(m):
        row = np.zeros(n_var)
        row[x * qy : (x + 1) * qy] = 1.0
        row[-1] = -1.0
        a_eq.append(row)
        b_eq.append(0.0)

    a_ub = []
    b_ub = []
    for y in range(qy):
        row = np.zeros(n_var)
        for x in range(m):
            row[x * qy + y] = 1.0
        row[-1] = -1.0
        a_ub.append(row)
        b_ub.append(0.0)

    bounds = [(0.0, float(channel.reshape(-1)[i])) for i in range(n_b)] + [(0.0, 1.0)]
    result = linprog(
        c,
        A_ub=np.asarray(a_ub),
        b_ub=np.asarray(b_ub),
        A_eq=np.asarray(a_eq),
        b_eq=np.asarray(b_eq),
        bounds=bounds,
        method="highs",
    )
    if not result.success:
        raise RuntimeError(f"Compact injective LP failed: {result.message}")
    return float(result.x[-1])


def sample_vertex_representations(
    channel: np.ndarray,
    rng: np.random.Generator,
    count: int,
    prefix: str = "VERTEX",
) -> list[Representation]:
    channel = np.asarray(channel, dtype=float)
    maps = enumerate_maps(*channel.shape)
    candidates: list[Representation] = []
    seen: set[tuple[tuple[tuple[int, ...], float], ...]] = set()
    deterministic_objectives = [
        np.arange(len(maps), dtype=float),
        -np.arange(len(maps), dtype=float),
        np.array([map_kappa(atom, channel.shape[1]) for atom in maps]),
        np.array([map_uniform_fiber_ambiguity(atom, channel.shape[1]) for atom in maps]),
    ]
    objectives = deterministic_objectives + [rng.normal(size=len(maps)) for _ in range(max(0, count - len(deterministic_objectives)))]
    for index, objective in enumerate(objectives[:count]):
        try:
            rep = solve_representation_lp(
                channel,
                objective,
                f"{prefix}_{index:03d}",
                maps=maps,
                metadata={"construction": "random_lp_vertex", "objective_index": index},
            )
        except RuntimeError:
            continue
        key = tuple(sorted((atom.outputs, round(float(weight), 11)) for atom, weight in zip(rep.maps, rep.weights, strict=True)))
        if key not in seen:
            candidates.append(rep)
            seen.add(key)
    return candidates


def bac_representations(a: float, b: float, grid_points: int = 9) -> list[Representation]:
    lower = max(0.0, b - a)
    upper = min(b, 1.0 - a)
    independent_t = (1.0 - a) * b
    values = set(np.linspace(lower, upper, grid_points).tolist())
    values.update([lower, upper, independent_t, (lower + upper) / 2.0])
    atoms = {
        "I": DeterministicMap((0, 1)),
        "F": DeterministicMap((1, 0)),
        "C0": DeterministicMap((0, 0)),
        "C1": DeterministicMap((1, 1)),
    }
    channel = np.array([[1.0 - a, a], [b, 1.0 - b]], dtype=float)
    reps: list[Representation] = []
    for index, t in enumerate(sorted(values)):
        weights = {
            "I": 1.0 - a - t,
            "F": b - t,
            "C0": t,
            "C1": a - b + t,
        }
        positive = [(key, value) for key, value in weights.items() if value > 1e-12]
        rep = Representation(
            name=f"BAC_t{t:.8f}",
            maps=[atoms[key] for key, _ in positive],
            weights=np.array([value for _, value in positive]),
            metadata={
                "construction": "complete_bac_interval_grid",
                "a": a,
                "b": b,
                "t": t,
                "is_independent_coupling": abs(t - independent_t) < 1e-10,
                "is_lower_endpoint": abs(t - lower) < 1e-10,
                "is_upper_endpoint": abs(t - upper) < 1e-10,
            },
        ).reduced()
        rep.verify(channel)
        reps.append(rep)
    return reps


def bec_representations(epsilon: float, grid_points: int = 7) -> list[Representation]:
    max_s = min(epsilon, 1.0 - epsilon)
    reps: list[Representation] = []
    channel = np.array([[1.0 - epsilon, 0.0, epsilon], [0.0, 1.0 - epsilon, epsilon]])
    base_maps = [
        DeterministicMap((0, 1)),
        DeterministicMap((0, 2)),
        DeterministicMap((2, 1)),
        DeterministicMap((2, 2)),
    ]
    for s in np.linspace(0.0, max_s, grid_points):
        weights = np.array([1.0 - epsilon - s, s, s, epsilon - s], dtype=float)
        rep = representation_from_lambda(
            f"BEC_s{s:.8f}",
            base_maps,
            weights,
            {"construction": "bec_one_parameter", "epsilon": epsilon, "s": float(s)},
        )
        rep.verify(channel)
        reps.append(rep)
    return reps


def unique_representations(reps: Iterable[Representation]) -> list[Representation]:
    result: list[Representation] = []
    seen: set[tuple[tuple[tuple[int, ...], float], ...]] = set()
    for rep in reps:
        reduced = rep.reduced()
        key = tuple(sorted((atom.outputs, round(float(weight), 11)) for atom, weight in zip(reduced.maps, reduced.weights, strict=True)))
        if key not in seen:
            seen.add(key)
            result.append(reduced)
    return result
