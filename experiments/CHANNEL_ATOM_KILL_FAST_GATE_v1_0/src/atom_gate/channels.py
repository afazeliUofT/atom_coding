from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations
from typing import Iterable

import numpy as np

from .models import DeterministicMap, Representation


@dataclass(frozen=True)
class ChannelSpec:
    name: str
    matrix: np.ndarray
    family: str
    nonadditive: bool
    metadata: dict[str, object]

    def __post_init__(self) -> None:
        matrix = np.asarray(self.matrix, dtype=float)
        object.__setattr__(self, "matrix", matrix)
        if matrix.ndim != 2:
            raise ValueError("Channel matrix must be 2-D")
        if np.any(matrix < -1e-12):
            raise ValueError("Channel entries must be nonnegative")
        if not np.allclose(matrix.sum(axis=1), 1.0, atol=1e-10):
            raise ValueError("Every channel row must sum to one")


def bsc(p: float) -> ChannelSpec:
    return ChannelSpec(
        name=f"BSC_p{p:.3f}",
        matrix=np.array([[1.0 - p, p], [p, 1.0 - p]], dtype=float),
        family="additive_control",
        nonadditive=False,
        metadata={"p": p},
    )


def bec(eps: float) -> ChannelSpec:
    return ChannelSpec(
        name=f"BEC_e{eps:.3f}",
        matrix=np.array([[1.0 - eps, 0.0, eps], [0.0, 1.0 - eps, eps]], dtype=float),
        family="erasure_control",
        nonadditive=True,
        metadata={"epsilon": eps},
    )


def bac(a: float, b: float, name: str | None = None) -> ChannelSpec:
    """Binary asymmetric channel: 0->1 with a, 1->0 with b."""
    matrix = np.array([[1.0 - a, a], [b, 1.0 - b]], dtype=float)
    return ChannelSpec(
        name=name or f"BAC_a{a:.3f}_b{b:.3f}",
        matrix=matrix,
        family="binary_asymmetric",
        nonadditive=not np.isclose(a, b),
        metadata={"a": a, "b": b},
    )


def asymmetric_erasure_stuck(
    p0: tuple[float, float, float],
    p1: tuple[float, float, float],
    name: str,
) -> ChannelSpec:
    return ChannelSpec(
        name=name,
        matrix=np.array([p0, p1], dtype=float),
        family="asymmetric_erasure_stuck",
        nonadditive=True,
        metadata={"row0": list(p0), "row1": list(p1)},
    )


def deterministic_control() -> ChannelSpec:
    return ChannelSpec(
        name="DETERMINISTIC_IDENTITY",
        matrix=np.eye(2, dtype=float),
        family="negative_control",
        nonadditive=False,
        metadata={},
    )


def useless_control() -> ChannelSpec:
    return ChannelSpec(
        name="USELESS_BINARY_OUTPUT",
        matrix=np.array([[0.35, 0.65], [0.35, 0.65]], dtype=float),
        family="negative_control",
        nonadditive=True,
        metadata={},
    )


def random_rational_channel(
    rng: np.random.Generator,
    input_size: int,
    output_size: int,
    denominator: int = 20,
    name: str | None = None,
) -> ChannelSpec:
    rows: list[np.ndarray] = []
    for _ in range(input_size):
        counts = rng.multinomial(denominator, np.full(output_size, 1.0 / output_size))
        while np.count_nonzero(counts) < min(2, output_size):
            counts = rng.multinomial(denominator, np.full(output_size, 1.0 / output_size))
        rows.append(counts / denominator)
    matrix = np.vstack(rows)
    return ChannelSpec(
        name=name or f"RAND_{input_size}x{output_size}_{rng.integers(1_000_000)}",
        matrix=matrix,
        family="random_rational",
        nonadditive=True,
        metadata={"denominator": denominator},
    )


def additive_qary_representation(q: int, weights: Iterable[float]) -> tuple[ChannelSpec, Representation]:
    probs = np.asarray(list(weights), dtype=float)
    if len(probs) > q:
        raise ValueError("At most q translation atoms are allowed")
    probs = probs / probs.sum()
    maps = [DeterministicMap(tuple((x + shift) % q for x in range(q))) for shift in range(len(probs))]
    rep = Representation("QARY_ADDITIVE", maps, probs)
    matrix = rep.induced_channel(q)
    return (
        ChannelSpec(
            name=f"QARY_ADDITIVE_q{q}",
            matrix=matrix,
            family="additive_control",
            nonadditive=False,
            metadata={"q": q, "weights": probs.tolist()},
        ),
        rep,
    )


def noncyclic_reversible_action_channel() -> tuple[ChannelSpec, Representation]:
    """A q=5 pairwise-disjoint permutation family not cyclic-circulant under relabeling."""
    maps = [
        DeterministicMap((0, 1, 2, 3, 4)),
        DeterministicMap((1, 0, 3, 4, 2)),
        DeterministicMap((2, 3, 4, 0, 1)),
    ]
    weights = np.array([0.67, 0.22, 0.11], dtype=float)
    rep = Representation("NONCYCLIC_REVERSIBLE_ACTION", maps, weights)
    matrix = rep.induced_channel(5)
    return (
        ChannelSpec(
            name="NONCYCLIC_REVERSIBLE_ACTION_q5",
            matrix=matrix,
            family="reversible_action",
            nonadditive=True,
            metadata={
                "q": 5,
                "weights": weights.tolist(),
                "claim": "not cyclic-circulant under row/output relabeling",
            },
        ),
        rep,
    )


def is_binary_additive(matrix: np.ndarray, atol: float = 1e-10) -> bool:
    matrix = np.asarray(matrix, dtype=float)
    return matrix.shape == (2, 2) and np.allclose(matrix[1], matrix[0][::-1], atol=atol)


def is_cyclic_circulant_up_to_relabeling(matrix: np.ndarray, atol: float = 1e-10) -> bool:
    matrix = np.asarray(matrix, dtype=float)
    if matrix.shape[0] != matrix.shape[1]:
        return False
    q = matrix.shape[0]
    for col_perm in permutations(range(q)):
        permuted = matrix[:, col_perm]
        base = permuted[0]
        shifts = [np.roll(base, shift) for shift in range(q)]
        assignments: list[int] = []
        possible = True
        for row in permuted:
            matches = [i for i, shifted in enumerate(shifts) if np.allclose(row, shifted, atol=atol)]
            if not matches:
                possible = False
                break
            assignments.append(matches[0])
        if possible and len(set(assignments)) == q:
            return True
    return False


def default_channel_suite(rng: np.random.Generator, random_count: int = 8) -> list[ChannelSpec]:
    channels: list[ChannelSpec] = [
        deterministic_control(),
        useless_control(),
        bsc(0.08),
        bsc(0.20),
        bec(0.25),
        bec(0.60),
        bac(0.10, 0.25),
        bac(0.20, 0.35),
        bac(0.30, 0.40),
        asymmetric_erasure_stuck((0.75, 0.05, 0.20), (0.10, 0.65, 0.25), "AES_INJECTIVE_FRIENDLY"),
        asymmetric_erasure_stuck((0.40, 0.05, 0.55), (0.10, 0.25, 0.65), "AES_OVERLOADED_ERASURE"),
    ]
    for i in range(random_count):
        channels.append(random_rational_channel(rng, 2, 3, denominator=20, name=f"RAND_2x3_{i:02d}"))
    for i in range(max(2, random_count // 3)):
        channels.append(random_rational_channel(rng, 3, 3, denominator=15, name=f"RAND_3x3_{i:02d}"))
    return channels
