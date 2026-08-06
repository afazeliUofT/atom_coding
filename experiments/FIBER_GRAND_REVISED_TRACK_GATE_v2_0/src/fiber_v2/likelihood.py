from __future__ import annotations

import itertools
import math
from typing import Sequence

from .channels import FixedDeletionBSC
from .utils import delete_positions


def component_probability(weight: int, m: int, p: float, stream_count: int) -> float:
    if p == 0.0:
        return (1.0 / stream_count) if weight == 0 else 0.0
    return (1.0 / stream_count) * (p**weight) * ((1.0 - p) ** (m - weight))


def deletion_likelihood(word: int, received: int, channel: FixedDeletionBSC) -> tuple[float, int]:
    total = 0.0
    operations = 0
    subsets = channel.deletion_subsets
    q = 1.0 / len(subsets)
    for positions in subsets:
        survivor = delete_positions(word, positions, channel.n)
        mismatches = (survivor ^ int(received)).bit_count()
        operations += channel.n + channel.m
        if channel.p == 0.0:
            total += q if mismatches == 0 else 0.0
        else:
            total += q * (channel.p**mismatches) * ((1.0 - channel.p) ** (channel.m - mismatches))
    return float(total), operations


def exhaustive_ml(codewords: Sequence[int], received: int, channel: FixedDeletionBSC, tolerance: float = 1e-13) -> tuple[tuple[int, ...], list[float], int]:
    scores: list[float] = []
    operations = 0
    for word in codewords:
        score, ops = deletion_likelihood(int(word), received, channel)
        scores.append(score)
        operations += ops
    best = max(scores)
    ties = tuple(i for i, score in enumerate(scores) if abs(score - best) <= tolerance)
    return ties, scores, operations
