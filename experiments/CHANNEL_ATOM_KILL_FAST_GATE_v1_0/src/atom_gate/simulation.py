from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .codes import BinaryLinearCode, QaryRandomCodebook, int_to_digits
from .models import Representation


def sample_channel_output(
    channel: np.ndarray,
    input_symbols: Sequence[int],
    rng: np.random.Generator,
) -> tuple[int, ...]:
    w = np.asarray(channel, dtype=float)
    outputs = []
    for x in input_symbols:
        outputs.append(int(rng.choice(w.shape[1], p=w[int(x)])))
    return tuple(outputs)


def sample_binary_linear_trial(
    channel: np.ndarray,
    code: BinaryLinearCode,
    rng: np.random.Generator,
) -> tuple[int, tuple[int, ...], tuple[int, ...]]:
    message = int(rng.integers(0, code.size))
    x = tuple(int(v) for v in code.codewords_array[message])
    y = sample_channel_output(channel, x, rng)
    return message, x, y


def sample_qary_trial(
    channel: np.ndarray,
    code: QaryRandomCodebook,
    rng: np.random.Generator,
) -> tuple[int, tuple[int, ...], tuple[int, ...]]:
    message = int(rng.integers(0, code.size))
    x = tuple(int(v) for v in code.digits[message])
    y = sample_channel_output(channel, x, rng)
    return message, x, y


def top_product_atoms(
    representation: Representation,
    blocklength: int,
    cumulative_target: float,
    max_atoms: int,
) -> tuple[Representation, list[tuple[int, ...]], float]:
    from .enumeration import ProductAtomEnumerator

    rep = representation.reduced().sorted_by_weight()
    enumerator = ProductAtomEnumerator(rep.weights, blocklength)
    atoms: list[tuple[int, ...]] = []
    cumulative = 0.0
    for item in enumerator:
        atoms.append(item.atom_indices)
        cumulative += item.probability
        if cumulative >= cumulative_target or len(atoms) >= max_atoms:
            break
    return rep, atoms, min(1.0, cumulative)
