from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np

from .channels import DeletionChannel, InsertionChannel
from .codes import BinaryCode, SystematicLinearCode, crc_systematic_code
from .likelihood import (
    deletion_likelihood_dp,
    one_deletion_mismatch_vector,
    one_insertion_mismatch_vector,
)


CRC_POLYNOMIALS = {
    3: 0b1011,
    4: 0b10011,
    5: 0b100101,
    6: 0b1000011,
    7: 0b10000011,
    8: 0b100000111,
}


def make_code(family: str, n: int, k: int, rng: np.random.Generator, label: str = "") -> BinaryCode:
    if family == "RLC":
        return SystematicLinearCode.random_systematic(n, k, rng, name=f"RLC{label}_n{n}_k{k}")
    if family == "CRC":
        degree = n - k
        polynomial = CRC_POLYNOMIALS.get(degree, (1 << degree) | 0b11)
        return crc_systematic_code(n, k, polynomial, name=f"CRC{label}_n{n}_k{k}")
    raise ValueError(f"Unknown code family {family}")


def deletion_history_components(word: int, received: int, channel: DeletionChannel) -> np.ndarray:
    if channel.deletions == 1:
        mismatches = one_deletion_mismatch_vector(word, received, channel.n)
        q = (
            np.full(channel.n, 1.0 / channel.n, dtype=float)
            if channel.deletion_weights is None
            else np.asarray(channel.deletion_weights, dtype=float)
        )
        p = channel.substitution_probability
        if p == 0.0:
            masses = (mismatches == 0).astype(float)
        else:
            masses = np.power(p, mismatches) * np.power(1.0 - p, channel.n - 1 - mismatches)
        return q * masses

    # Enumerate deletion subsets only for diagnostic small t.
    import itertools
    from .utils import binary_tuple

    x = binary_tuple(word, channel.n)
    y = binary_tuple(received, channel.n - channel.deletions)
    subsets = list(itertools.combinations(range(channel.n), channel.deletions))
    q = 1.0 / len(subsets)
    values = []
    p = channel.substitution_probability
    for deleted in subsets:
        deleted_set = set(deleted)
        survivors = tuple(x[index] for index in range(channel.n) if index not in deleted_set)
        d = sum(int(a != b) for a, b in zip(survivors, y))
        value = q * ((d == 0) if p == 0.0 else p**d * (1.0 - p) ** (channel.n - channel.deletions - d))
        values.append(float(value))
    return np.asarray(values, dtype=float)


def insertion_history_components(word: int, received: int, channel: InsertionChannel) -> np.ndarray:
    mismatches = one_insertion_mismatch_vector(word, received, channel.n)
    q = (
        np.full(channel.n + 1, 1.0 / (channel.n + 1), dtype=float)
        if channel.insertion_position_weights is None
        else np.asarray(channel.insertion_position_weights, dtype=float)
    )
    y = tuple((received >> i) & 1 for i in range(channel.n + 1))
    p = channel.substitution_probability
    values = []
    for j, d in enumerate(mismatches):
        inserted_mass = channel.inserted_one_probability if y[j] else 1.0 - channel.inserted_one_probability
        bsc = (d == 0) if p == 0.0 else p**int(d) * (1.0 - p) ** (channel.n - int(d))
        values.append(float(q[j] * inserted_mass * bsc))
    return np.asarray(values, dtype=float)


def effective_history_count(components: Sequence[float]) -> float:
    values = np.asarray(components, dtype=float)
    total = float(np.sum(values))
    square = float(np.sum(values * values))
    if total <= 0.0 or square <= 0.0:
        return 0.0
    return total * total / square


def history_entropy_bits(components: Sequence[float]) -> float:
    values = np.asarray(components, dtype=float)
    total = float(np.sum(values))
    if total <= 0.0:
        return 0.0
    probabilities = values[values > 0] / total
    return float(-np.sum(probabilities * np.log2(probabilities)))


def result_row_base(
    code: BinaryCode,
    channel_name: str,
    n: int,
    trial: int,
    true_message: int,
    ml_tie_set: Sequence[int],
) -> dict[str, Any]:
    return {
        "code": code.name,
        "code_family": code.family,
        "code_size": code.size,
        "code_rate": code.rate,
        "channel": channel_name,
        "n": n,
        "trial": trial,
        "true_message": int(true_message),
        "ml_contains_true": int(true_message) in set(int(v) for v in ml_tie_set),
        "ml_tie_size": len(tuple(ml_tie_set)),
    }
