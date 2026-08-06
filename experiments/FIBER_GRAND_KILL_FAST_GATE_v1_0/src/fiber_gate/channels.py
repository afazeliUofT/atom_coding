from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from .utils import binary_tuple, tuple_to_int


@dataclass(frozen=True)
class DeletionChannel:
    n: int
    deletions: int
    substitution_probability: float
    deletion_weights: tuple[float, ...] | None = None
    name: str | None = None

    def __post_init__(self) -> None:
        if not (1 <= self.deletions < self.n):
            raise ValueError("Require 1 <= deletions < n")
        if not (0.0 <= self.substitution_probability < 0.5):
            raise ValueError("Substitution probability must be in [0, 1/2)")
        if self.deletion_weights is not None and self.deletions != 1:
            raise ValueError("Nonuniform weights are implemented only for one deletion")
        if self.deletion_weights is not None:
            weights = np.asarray(self.deletion_weights, dtype=float)
            if len(weights) != self.n or np.any(weights < 0) or not np.isclose(weights.sum(), 1.0):
                raise ValueError("Deletion weights must be a probability vector of length n")

    @property
    def output_length(self) -> int:
        return self.n - self.deletions

    @property
    def label(self) -> str:
        return self.name or f"D{self.deletions}_BSC_p{self.substitution_probability:.3f}_n{self.n}"

    def metadata(self) -> dict[str, Any]:
        return {
            "name": self.label,
            "n": self.n,
            "deletions": self.deletions,
            "substitution_probability": self.substitution_probability,
            "deletion_weights": None if self.deletion_weights is None else list(self.deletion_weights),
        }


@dataclass(frozen=True)
class InsertionChannel:
    n: int
    substitution_probability: float
    insertion_position_weights: tuple[float, ...] | None = None
    inserted_one_probability: float = 0.5
    name: str | None = None

    def __post_init__(self) -> None:
        if not (0.0 <= self.substitution_probability < 0.5):
            raise ValueError("Substitution probability must be in [0, 1/2)")
        if not (0.0 < self.inserted_one_probability < 1.0):
            raise ValueError("Inserted bit law must have both symbols positive")
        if self.insertion_position_weights is not None:
            weights = np.asarray(self.insertion_position_weights, dtype=float)
            if len(weights) != self.n + 1 or np.any(weights < 0) or not np.isclose(weights.sum(), 1.0):
                raise ValueError("Insertion-position weights must have length n+1")

    @property
    def output_length(self) -> int:
        return self.n + 1

    @property
    def label(self) -> str:
        return self.name or f"I1_BSC_p{self.substitution_probability:.3f}_n{self.n}"

    def metadata(self) -> dict[str, Any]:
        return {
            "name": self.label,
            "n": self.n,
            "substitution_probability": self.substitution_probability,
            "position_weights": None if self.insertion_position_weights is None else list(self.insertion_position_weights),
            "inserted_one_probability": self.inserted_one_probability,
        }


def edge_biased_weights(length: int, edge_mass: float = 0.55) -> tuple[float, ...]:
    if length < 2:
        return (1.0,)
    remaining = 1.0 - edge_mass
    weights = np.full(length, remaining / max(1, length - 2), dtype=float)
    if length == 2:
        weights[:] = 0.5
    else:
        weights[0] = edge_mass / 2.0
        weights[-1] = edge_mass / 2.0
    weights /= weights.sum()
    return tuple(float(v) for v in weights)


def sample_deletion_channel(
    word: int,
    channel: DeletionChannel,
    rng: np.random.Generator,
) -> tuple[int, tuple[int, ...], int]:
    bits = np.asarray(binary_tuple(word, channel.n), dtype=np.uint8)
    substitutions = rng.binomial(1, channel.substitution_probability, size=channel.n).astype(np.uint8)
    corrupted = bits ^ substitutions
    if channel.deletions == 1 and channel.deletion_weights is not None:
        weights = np.asarray(channel.deletion_weights, dtype=float)
        deleted = (int(rng.choice(channel.n, p=weights)),)
    else:
        deleted = tuple(sorted(int(v) for v in rng.choice(channel.n, size=channel.deletions, replace=False)))
    received = tuple(int(corrupted[index]) for index in range(channel.n) if index not in set(deleted))
    return tuple_to_int(received), deleted, tuple_to_int(substitutions)


def sample_insertion_channel(
    word: int,
    channel: InsertionChannel,
    rng: np.random.Generator,
) -> tuple[int, int, int, int]:
    bits = np.asarray(binary_tuple(word, channel.n), dtype=np.uint8)
    substitutions = rng.binomial(1, channel.substitution_probability, size=channel.n).astype(np.uint8)
    corrupted = bits ^ substitutions
    if channel.insertion_position_weights is None:
        position = int(rng.integers(0, channel.n + 1))
    else:
        position = int(rng.choice(channel.n + 1, p=np.asarray(channel.insertion_position_weights, dtype=float)))
    inserted = int(rng.random() < channel.inserted_one_probability)
    output = list(int(v) for v in corrupted)
    output.insert(position, inserted)
    return tuple_to_int(output), position, inserted, tuple_to_int(substitutions)
