from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable, Sequence

import numpy as np


def _as_bits(sequence: Sequence[int] | np.ndarray | str) -> tuple[int, ...]:
    if isinstance(sequence, str):
        bits = tuple(int(ch) for ch in sequence.strip())
    else:
        bits = tuple(int(v) for v in sequence)
    if any(bit not in (0, 1) for bit in bits):
        raise ValueError("Only binary sequences are supported")
    return bits


@dataclass(frozen=True)
class LZ78Record:
    parent: int
    symbol: int  # 0/1, or 2 for EOF


def lz78_records(sequence: Sequence[int] | np.ndarray | str) -> tuple[LZ78Record, ...]:
    """Canonical fixed-block LZ78 parse with an explicit EOF record.

    A record at phrase number r uses a parent index in {0,...,r-1} and a
    ternary tag {0,1,EOF}. The root phrase has index 0. New phrases are added
    only for 0/1 records; EOF emits an existing final phrase. This is an
    injective encoder for a fixed blocklength n.
    """
    bits = _as_bits(sequence)
    dictionary: dict[tuple[int, ...], int] = {(): 0}
    records: list[LZ78Record] = []
    position = 0
    while position < len(bits):
        phrase: tuple[int, ...] = ()
        index = 0
        cursor = position
        while cursor < len(bits) and phrase + (bits[cursor],) in dictionary:
            phrase = phrase + (bits[cursor],)
            index = dictionary[phrase]
            cursor += 1
        if cursor < len(bits):
            symbol = bits[cursor]
            records.append(LZ78Record(index, symbol))
            new_phrase = phrase + (symbol,)
            dictionary[new_phrase] = len(dictionary)
            position = cursor + 1
        else:
            records.append(LZ78Record(index, 2))
            position = cursor
    if len(bits) == 0:
        records.append(LZ78Record(0, 2))
    return tuple(records)


def lz78_decode(records: Iterable[LZ78Record], blocklength: int) -> tuple[int, ...]:
    dictionary: list[tuple[int, ...]] = [()]
    output: list[int] = []
    eof_seen = False
    for record in records:
        if eof_seen:
            raise ValueError("Records after EOF")
        if record.parent < 0 or record.parent >= len(dictionary):
            raise ValueError("Invalid parent index")
        parent = dictionary[record.parent]
        if record.symbol in (0, 1):
            phrase = parent + (record.symbol,)
            output.extend(phrase)
            dictionary.append(phrase)
        elif record.symbol == 2:
            output.extend(parent)
            eof_seen = True
        else:
            raise ValueError("Invalid LZ78 tag")
        if len(output) > blocklength:
            raise ValueError("Decoded sequence exceeds blocklength")
    if len(output) != blocklength:
        raise ValueError("Decoded sequence has wrong blocklength")
    return tuple(output)


def lz78_codelength_bits(sequence: Sequence[int] | np.ndarray | str) -> int:
    """Length of the canonical fixed-block LZ78 record stream.

    At record r (1-indexed), the parent index needs ceil(log2 r) bits because
    the dictionary contains indices 0,...,r-1. The 0/1/EOF tag uses two bits.
    The blocklength is assumed known to the decoder and is not charged.
    """
    records = lz78_records(sequence)
    total = 0
    for r, _ in enumerate(records, start=1):
        width = 0 if r <= 1 else math.ceil(math.log2(r))
        total += width + 2
    return int(total)


def lz78_phrase_count(sequence: Sequence[int] | np.ndarray | str) -> int:
    return len(lz78_records(sequence))


def _log_beta_half_counts(n0: float, n1: float) -> float:
    return (
        math.lgamma(n0 + 0.5)
        + math.lgamma(n1 + 0.5)
        - math.lgamma(n0 + n1 + 1.0)
        - 2.0 * math.lgamma(0.5)
    )


def kt_log2_probability(sequence: Sequence[int] | np.ndarray | str, order: int = 0, initial_context: int = 0) -> float:
    """Binary Krichevsky-Trofimov log probability for a fixed Markov order.

    Contexts are initialized by `order` copies of the least-significant bits of
    initial_context. For order zero, this is the ordinary KT mixture.
    """
    bits = _as_bits(sequence)
    if order < 0 or order > 12:
        raise ValueError("Supported KT order is 0..12")
    contexts = 1 << order
    counts0 = np.zeros(contexts, dtype=float)
    counts1 = np.zeros(contexts, dtype=float)
    if order == 0:
        context = 0
        mask = 0
    else:
        context = int(initial_context) & (contexts - 1)
        mask = contexts - 1
    logp = 0.0
    for bit in bits:
        n0 = counts0[context]
        n1 = counts1[context]
        denominator = n0 + n1 + 1.0
        numerator = (n1 + 0.5) if bit else (n0 + 0.5)
        logp += math.log2(numerator / denominator)
        if bit:
            counts1[context] += 1.0
        else:
            counts0[context] += 1.0
        if order:
            context = ((context << 1) | bit) & mask
    return float(logp)


def kt_codelength_bits(sequence: Sequence[int] | np.ndarray | str, order: int = 0, initial_context: int = 0) -> float:
    return -kt_log2_probability(sequence, order=order, initial_context=initial_context)


@dataclass
class _CTWNode:
    depth: int
    max_depth: int
    n0: int = 0
    n1: int = 0
    child0: "_CTWNode | None" = None
    child1: "_CTWNode | None" = None
    log_kt: float = 0.0
    log_weighted: float = 0.0

    def child(self, bit: int) -> "_CTWNode":
        if bit == 0:
            if self.child0 is None:
                self.child0 = _CTWNode(self.depth + 1, self.max_depth)
            return self.child0
        if self.child1 is None:
            self.child1 = _CTWNode(self.depth + 1, self.max_depth)
        return self.child1

    def recompute(self) -> None:
        self.log_kt = _log_beta_half_counts(self.n0, self.n1)
        if self.depth >= self.max_depth:
            self.log_weighted = self.log_kt
            return
        child_sum = 0.0
        if self.child0 is not None:
            child_sum += self.child0.log_weighted
        if self.child1 is not None:
            child_sum += self.child1.log_weighted
        a = math.log(0.5) + self.log_kt
        b = math.log(0.5) + child_sum
        maximum = max(a, b)
        self.log_weighted = maximum + math.log(math.exp(a - maximum) + math.exp(b - maximum))


class BinaryCTW:
    """Exact full-tree binary CTW probability with a fixed padded context."""

    def __init__(self, depth: int, initial_context: Sequence[int] | None = None) -> None:
        if depth < 0 or depth > 16:
            raise ValueError("CTW depth must be 0..16")
        self.depth = int(depth)
        if initial_context is None:
            self.history = [0] * depth
        else:
            context = [int(v) for v in initial_context]
            if len(context) != depth or any(v not in (0, 1) for v in context):
                raise ValueError("Initial context must contain depth binary symbols")
            self.history = context
        self.root = _CTWNode(0, depth)
        self.log_probability_nats = 0.0

    def _path(self) -> list[_CTWNode]:
        nodes = [self.root]
        node = self.root
        # Most recent symbol first is the standard suffix-tree direction.
        for bit in reversed(self.history[-self.depth :]):
            node = node.child(bit)
            nodes.append(node)
        return nodes

    def update(self, bit: int) -> float:
        bit = int(bit)
        if bit not in (0, 1):
            raise ValueError("Binary symbol required")
        before = self.root.log_weighted
        nodes = self._path()
        for node in nodes:
            if bit:
                node.n1 += 1
            else:
                node.n0 += 1
        for node in reversed(nodes):
            node.recompute()
        after = self.root.log_weighted
        self.log_probability_nats = after
        if self.depth:
            self.history.append(bit)
            self.history = self.history[-self.depth :]
        return float((after - before) / math.log(2.0))

    def log2_probability(self, sequence: Sequence[int] | np.ndarray | str) -> float:
        total = 0.0
        for bit in _as_bits(sequence):
            total += self.update(bit)
        return float(total)


def ctw_log2_probability(sequence: Sequence[int] | np.ndarray | str, depth: int = 2) -> float:
    model = BinaryCTW(depth)
    return model.log2_probability(sequence)


def ctw_codelength_bits(sequence: Sequence[int] | np.ndarray | str, depth: int = 2) -> float:
    return -ctw_log2_probability(sequence, depth=depth)


def markov_log2_probability(
    sequence: Sequence[int] | np.ndarray | str,
    p01: float,
    p10: float,
    initial_probability_one: float | None = None,
) -> float:
    bits = _as_bits(sequence)
    if not bits:
        return 0.0
    if not (0.0 < p01 < 1.0 and 0.0 < p10 < 1.0):
        raise ValueError("Transition probabilities must lie strictly between zero and one")
    if initial_probability_one is None:
        initial_probability_one = p01 / (p01 + p10)
    pi1 = min(max(float(initial_probability_one), 1e-15), 1.0 - 1e-15)
    logp = math.log2(pi1 if bits[0] else 1.0 - pi1)
    for previous, current in zip(bits, bits[1:]):
        if previous == 0:
            probability = p01 if current else 1.0 - p01
        else:
            probability = p10 if current == 0 else 1.0 - p10
        logp += math.log2(probability)
    return float(logp)


def empirical_markov_log2_probability(sequence: Sequence[int] | np.ndarray | str) -> float:
    bits = _as_bits(sequence)
    if not bits:
        return 0.0
    counts = [[0, 0], [0, 0]]
    for a, b in zip(bits, bits[1:]):
        counts[a][b] += 1
    logp = -1.0  # fixed uniform initial symbol
    for state in (0, 1):
        total = counts[state][0] + counts[state][1]
        if total == 0:
            continue
        for symbol in (0, 1):
            count = counts[state][symbol]
            if count:
                logp += count * math.log2(count / total)
    return float(logp)


def renyi_half_rate_markov(p01: float, p10: float) -> float:
    matrix = np.array(
        [[math.sqrt(1.0 - p01), math.sqrt(p01)], [math.sqrt(p10), math.sqrt(1.0 - p10)]],
        dtype=float,
    )
    spectral_radius = float(np.max(np.abs(np.linalg.eigvals(matrix))))
    return 2.0 * math.log2(spectral_radius)


def entropy_rate_markov(p01: float, p10: float) -> float:
    pi1 = p01 / (p01 + p10)
    pi0 = 1.0 - pi1
    def h2(p: float) -> float:
        if p <= 0.0 or p >= 1.0:
            return 0.0
        return -p * math.log2(p) - (1.0 - p) * math.log2(1.0 - p)
    return pi0 * h2(p01) + pi1 * h2(p10)
