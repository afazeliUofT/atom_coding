from __future__ import annotations

import itertools
import math
from fractions import Fraction
from typing import Sequence

import numpy as np

from .channels import DeletionChannel, InsertionChannel
from .utils import binary_tuple, int_array


def _bernoulli_mass(mismatches: int, total: int, p: float) -> float:
    if p == 0.0:
        return 1.0 if mismatches == 0 else 0.0
    return float((p ** mismatches) * ((1.0 - p) ** (total - mismatches)))


def _bernoulli_mass_fraction(mismatches: int, total: int, p: Fraction) -> Fraction:
    if p == 0:
        return Fraction(1, 1) if mismatches == 0 else Fraction(0, 1)
    return (p ** mismatches) * ((1 - p) ** (total - mismatches))


def one_deletion_mismatch_vector(word: int, received: int, n: int) -> np.ndarray:
    x = binary_tuple(word, n)
    y = binary_tuple(received, n - 1)
    d = sum(int(x[index] != y[index - 1]) for index in range(1, n))
    values = [d]
    for j in range(n - 1):
        d = d - int(x[j + 1] != y[j]) + int(x[j] != y[j])
        values.append(d)
    return np.asarray(values, dtype=np.int16)


def one_deletion_likelihood(word: int, received: int, channel: DeletionChannel) -> float:
    if channel.deletions != 1:
        raise ValueError("Channel must have exactly one deletion")
    mismatches = one_deletion_mismatch_vector(word, received, channel.n)
    if channel.deletion_weights is None:
        weights = np.full(channel.n, 1.0 / channel.n, dtype=float)
    else:
        weights = np.asarray(channel.deletion_weights, dtype=float)
    masses = np.asarray(
        [_bernoulli_mass(int(d), channel.n - 1, channel.substitution_probability) for d in mismatches],
        dtype=float,
    )
    return float(np.dot(weights, masses))


def one_deletion_likelihood_fraction(
    word: int,
    received: int,
    n: int,
    p: Fraction,
    weights: Sequence[Fraction] | None = None,
) -> Fraction:
    mismatches = one_deletion_mismatch_vector(word, received, n)
    q = list(weights) if weights is not None else [Fraction(1, n)] * n
    return sum(
        q[j] * _bernoulli_mass_fraction(int(mismatches[j]), n - 1, p)
        for j in range(n)
    )


def deletion_likelihood_dp(word: int, received: int, channel: DeletionChannel) -> float:
    """Exact likelihood for exactly t uniform deletions and BSC substitutions.

    The dynamic program has O(n t) states because after processing i input
    symbols the output index equals i minus the number of deletions used.
    """
    if channel.deletions == 1 and channel.deletion_weights is not None:
        return one_deletion_likelihood(word, received, channel)
    n = channel.n
    t = channel.deletions
    y = binary_tuple(received, n - t)
    x = binary_tuple(word, n)
    dp = np.zeros(t + 1, dtype=float)
    dp[0] = 1.0
    for i in range(n):
        nxt = np.zeros_like(dp)
        for d in range(min(t, i) + 1):
            mass = float(dp[d])
            if mass == 0.0:
                continue
            if d < t:
                nxt[d + 1] += mass
            output_index = i - d
            if output_index < n - t:
                mismatch = int(x[i] != y[output_index])
                factor = channel.substitution_probability if mismatch else 1.0 - channel.substitution_probability
                nxt[d] += mass * factor
        dp = nxt
    return float(dp[t] / math.comb(n, t))


def deletion_likelihood_vectorized(
    codewords_array: np.ndarray,
    received: int,
    channel: DeletionChannel,
) -> tuple[np.ndarray, int]:
    codewords = np.asarray(codewords_array, dtype=np.uint8)
    if codewords.shape[1] != channel.n:
        raise ValueError("Codeword length mismatch")
    m = len(codewords)
    n = channel.n
    t = channel.deletions
    y = np.asarray(binary_tuple(received, n - t), dtype=np.uint8)
    if t == 1 and channel.deletion_weights is not None:
        # Exact O(M n) recurrence for arbitrary deletion-position weights.
        weights = np.asarray(channel.deletion_weights, dtype=float)
        d = np.count_nonzero(codewords[:, 1:] != y[None, :], axis=1).astype(np.int16)
        scores = weights[0] * np.asarray(
            [_bernoulli_mass(int(v), n - 1, channel.substitution_probability) for v in d],
            dtype=float,
        )
        operations = m * (n - 1)
        current = d
        for j in range(n - 1):
            current = current - (codewords[:, j + 1] != y[j]).astype(np.int16) + (codewords[:, j] != y[j]).astype(np.int16)
            if channel.substitution_probability == 0.0:
                mass = (current == 0).astype(float)
            else:
                p = channel.substitution_probability
                mass = np.power(p, current) * np.power(1.0 - p, n - 1 - current)
            scores += weights[j + 1] * mass
            operations += 4 * m
        return scores, int(operations)

    dp = np.zeros((m, t + 1), dtype=float)
    dp[:, 0] = 1.0
    operations = 0
    for i in range(n):
        nxt = np.zeros_like(dp)
        for d in range(min(t, i) + 1):
            if d < t:
                nxt[:, d + 1] += dp[:, d]
                operations += m
            output_index = i - d
            if output_index < n - t:
                mismatch = codewords[:, i] != y[output_index]
                factor = np.where(mismatch, channel.substitution_probability, 1.0 - channel.substitution_probability)
                nxt[:, d] += dp[:, d] * factor
                operations += 3 * m
        dp = nxt
    return dp[:, t] / math.comb(n, t), int(operations)


def one_insertion_mismatch_vector(word: int, received: int, n: int) -> np.ndarray:
    x = binary_tuple(word, n)
    y = binary_tuple(received, n + 1)
    values = []
    for j in range(n + 1):
        candidate = y[:j] + y[j + 1 :]
        values.append(sum(int(a != b) for a, b in zip(x, candidate)))
    return np.asarray(values, dtype=np.int16)


def one_insertion_likelihood(word: int, received: int, channel: InsertionChannel) -> float:
    mismatches = one_insertion_mismatch_vector(word, received, channel.n)
    if channel.insertion_position_weights is None:
        q = np.full(channel.n + 1, 1.0 / (channel.n + 1), dtype=float)
    else:
        q = np.asarray(channel.insertion_position_weights, dtype=float)
    y = binary_tuple(received, channel.n + 1)
    score = 0.0
    for j, d in enumerate(mismatches):
        insert_mass = channel.inserted_one_probability if y[j] else 1.0 - channel.inserted_one_probability
        score += q[j] * insert_mass * _bernoulli_mass(int(d), channel.n, channel.substitution_probability)
    return float(score)


def insertion_likelihood_vectorized(
    codewords_array: np.ndarray,
    received: int,
    channel: InsertionChannel,
) -> tuple[np.ndarray, int]:
    codewords = np.asarray(codewords_array, dtype=np.uint8)
    n = channel.n
    y = np.asarray(binary_tuple(received, n + 1), dtype=np.uint8)
    q = (
        np.full(n + 1, 1.0 / (n + 1), dtype=float)
        if channel.insertion_position_weights is None
        else np.asarray(channel.insertion_position_weights, dtype=float)
    )
    scores = np.zeros(len(codewords), dtype=float)
    operations = 0
    for j in range(n + 1):
        aligned = np.concatenate([y[:j], y[j + 1 :]])
        d = np.count_nonzero(codewords != aligned[None, :], axis=1)
        if channel.substitution_probability == 0.0:
            mass = (d == 0).astype(float)
        else:
            p = channel.substitution_probability
            mass = np.power(p, d) * np.power(1.0 - p, n - d)
        insert_mass = channel.inserted_one_probability if y[j] else 1.0 - channel.inserted_one_probability
        scores += q[j] * insert_mass * mass
        operations += len(codewords) * (n + 4)
    return scores, int(operations)


def exact_ml_indices(scores: np.ndarray, tolerance: float = 1e-12) -> tuple[int, ...]:
    maximum = float(np.max(scores))
    return tuple(int(v) for v in np.flatnonzero(scores >= maximum - tolerance))
