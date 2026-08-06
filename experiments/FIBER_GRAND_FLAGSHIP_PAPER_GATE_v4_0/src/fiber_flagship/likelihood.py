from __future__ import annotations

from typing import Sequence

from .channels import FixedDeletionBSC
from .utils import binary_tuple, delete_positions


def component_probability(weight: int, m: int, p: float, stream_count: int) -> float:
    if p == 0.0:
        return (1.0 / stream_count) if weight == 0 else 0.0
    return (1.0 / stream_count) * (p**weight) * ((1.0 - p) ** (m - weight))


def one_deletion_mismatch_counts(word: int, received: int, n: int) -> list[int]:
    x = binary_tuple(word, n)
    y = binary_tuple(received, n - 1)
    d = sum(int(x[i] != y[i - 1]) for i in range(1, n))
    counts = [d]
    for j in range(n - 1):
        d = d - int(x[j + 1] != y[j]) + int(x[j] != y[j])
        counts.append(d)
    return counts


def deletion_likelihood(word: int, received: int, channel: FixedDeletionBSC) -> tuple[float, int]:
    if channel.deletions == 1:
        counts = one_deletion_mismatch_counts(word, received, channel.n)
        if channel.p == 0.0:
            total = sum(1.0 for d in counts if d == 0) / channel.n
        else:
            p = channel.p
            total = sum((p**d) * ((1.0 - p) ** (channel.m - d)) for d in counts) / channel.n
        return float(total), 4 * channel.n

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


def exhaustive_ml(
    codewords: Sequence[int],
    received: int,
    channel: FixedDeletionBSC,
    tolerance: float = 1e-13,
) -> tuple[tuple[int, ...], list[float], int]:
    scores: list[float] = []
    operations = 0
    for word in codewords:
        score, ops = deletion_likelihood(int(word), received, channel)
        scores.append(score)
        operations += ops
    best = max(scores)
    ties = tuple(i for i, score in enumerate(scores) if abs(score - best) <= tolerance)
    return ties, scores, operations
