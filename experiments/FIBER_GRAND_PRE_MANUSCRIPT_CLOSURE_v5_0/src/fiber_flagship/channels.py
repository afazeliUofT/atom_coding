from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .utils import delete_positions


@dataclass(frozen=True)
class FixedDeletionBSC:
    n: int
    deletions: int
    p: float

    def __post_init__(self) -> None:
        if not (1 <= self.deletions < self.n):
            raise ValueError("require 1 <= deletions < n")
        if not (0.0 <= self.p < 0.5):
            raise ValueError("require 0 <= p < 1/2")

    @property
    def m(self) -> int:
        return self.n - self.deletions

    @property
    def deletion_subsets(self) -> tuple[tuple[int, ...], ...]:
        return tuple(itertools.combinations(range(self.n), self.deletions))

    @property
    def stream_count(self) -> int:
        import math

        return math.comb(self.n, self.deletions)

    @property
    def label(self) -> str:
        return f"D{self.deletions}_BSC_p{self.p:.3f}"


def sample_channel(word: int, channel: FixedDeletionBSC, rng: np.random.Generator) -> tuple[int, tuple[int, ...], int]:
    subsets = channel.deletion_subsets
    positions = subsets[int(rng.integers(0, len(subsets)))]
    survivor = delete_positions(word, positions, channel.n)
    error = 0
    for i in range(channel.m):
        if rng.random() < channel.p:
            error |= 1 << i
    return survivor ^ error, positions, error
