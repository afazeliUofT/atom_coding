from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable, Sequence

import numpy as np
from scipy.special import gammaln

from .utils import logsumexp2


@dataclass(frozen=True)
class BinaryMarkovType:
    n: int
    start: int
    end: int
    n00: int
    n01: int
    n10: int
    n11: int
    n0: int
    n1: int
    log2_count: float

    @property
    def key(self) -> tuple[int, int, int, int, int]:
        return (self.start, self.n00, self.n01, self.n10, self.n11)


def transition_counts(sequence: Sequence[int]) -> tuple[int, int, int, int, int]:
    bits = tuple(int(v) for v in sequence)
    if not bits:
        raise ValueError("Sequence must be nonempty")
    n00 = n01 = n10 = n11 = 0
    for a, b in zip(bits, bits[1:]):
        if a == 0 and b == 0:
            n00 += 1
        elif a == 0 and b == 1:
            n01 += 1
        elif a == 1 and b == 0:
            n10 += 1
        elif a == 1 and b == 1:
            n11 += 1
        else:
            raise ValueError("Binary sequence required")
    return bits[0], n00, n01, n10, n11


def _log2_comb(n: int, k: int) -> float:
    if k < 0 or k > n:
        return float("-inf")
    return (math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)) / math.log(2.0)


def _type_count_log2(n0: int, n1: int, r0: int, r1: int) -> float:
    if r0 == 0:
        return 0.0 if n0 == 0 and r1 == 1 and n1 > 0 else float("-inf")
    if r1 == 0:
        return 0.0 if n1 == 0 and r0 == 1 and n0 > 0 else float("-inf")
    if n0 < r0 or n1 < r1:
        return float("-inf")
    return _log2_comb(n0 - 1, r0 - 1) + _log2_comb(n1 - 1, r1 - 1)


@lru_cache(maxsize=64)
def enumerate_binary_markov_types(n: int) -> tuple[BinaryMarkovType, ...]:
    if n < 1:
        raise ValueError("n must be positive")
    if n == 1:
        return (
            BinaryMarkovType(1, 0, 0, 0, 0, 0, 0, 1, 0, 0.0),
            BinaryMarkovType(1, 1, 1, 0, 0, 0, 0, 0, 1, 0.0),
        )
    types: list[BinaryMarkovType] = []
    # Parameterize by start/end and numbers of symbol runs.
    for start in (0, 1):
        for end in (0, 1):
            if start == 0 and end == 0:
                for cross in range(0, (n - 1) // 2 + 1):
                    r1 = cross
                    r0 = cross + 1
                    n01 = n10 = cross
                    min_n0, min_n1 = r0, r1
                    for n0 in range(min_n0, n - min_n1 + 1):
                        n1 = n - n0
                        if r1 == 0 and n1 != 0:
                            continue
                        log_count = _type_count_log2(n0, n1, r0, r1)
                        if math.isfinite(log_count):
                            types.append(BinaryMarkovType(n, start, end, n0-r0, n01, n10, n1-r1, n0, n1, log_count))
            elif start == 1 and end == 1:
                for cross in range(0, (n - 1) // 2 + 1):
                    r0 = cross
                    r1 = cross + 1
                    n01 = n10 = cross
                    min_n0, min_n1 = r0, r1
                    for n0 in range(min_n0, n - min_n1 + 1):
                        n1 = n - n0
                        if r0 == 0 and n0 != 0:
                            continue
                        log_count = _type_count_log2(n0, n1, r0, r1)
                        if math.isfinite(log_count):
                            types.append(BinaryMarkovType(n, start, end, n0-r0, n01, n10, n1-r1, n0, n1, log_count))
            elif start == 0 and end == 1:
                # r0=r1=r>=1; n01=r, n10=r-1.
                for runs in range(1, n // 2 + 1):
                    r0 = r1 = runs
                    n01, n10 = runs, runs - 1
                    for n0 in range(r0, n - r1 + 1):
                        n1 = n - n0
                        log_count = _type_count_log2(n0, n1, r0, r1)
                        if math.isfinite(log_count):
                            types.append(BinaryMarkovType(n, start, end, n0-r0, n01, n10, n1-r1, n0, n1, log_count))
            else:  # start 1, end 0
                for runs in range(1, n // 2 + 1):
                    r0 = r1 = runs
                    n10, n01 = runs, runs - 1
                    for n0 in range(r0, n - r1 + 1):
                        n1 = n - n0
                        log_count = _type_count_log2(n0, n1, r0, r1)
                        if math.isfinite(log_count):
                            types.append(BinaryMarkovType(n, start, end, n0-r0, n01, n10, n1-r1, n0, n1, log_count))
    # Deduplicate defensively and ensure total cardinality is 2^n.
    by_key = {item.key: item for item in types}
    ordered = tuple(sorted(by_key.values(), key=lambda item: item.key))
    total_log = logsumexp2([item.log2_count for item in ordered])
    if not math.isclose(total_log, float(n), rel_tol=0.0, abs_tol=2e-9):
        raise AssertionError(f"Markov type enumeration cardinality log2={total_log}, expected {n}")
    return ordered


@dataclass
class MarkovTypeAtlas:
    n: int
    types: tuple[BinaryMarkovType, ...]
    key_to_index: dict[tuple[int, int, int, int, int], int]
    starts: np.ndarray
    ends: np.ndarray
    n00: np.ndarray
    n01: np.ndarray
    n10: np.ndarray
    n11: np.ndarray
    n0: np.ndarray
    n1: np.ndarray
    log2_counts: np.ndarray

    @classmethod
    def build(cls, n: int) -> "MarkovTypeAtlas":
        types = enumerate_binary_markov_types(n)
        return cls(
            n=n,
            types=types,
            key_to_index={item.key: i for i, item in enumerate(types)},
            starts=np.array([item.start for item in types], dtype=np.int8),
            ends=np.array([item.end for item in types], dtype=np.int8),
            n00=np.array([item.n00 for item in types], dtype=np.int32),
            n01=np.array([item.n01 for item in types], dtype=np.int32),
            n10=np.array([item.n10 for item in types], dtype=np.int32),
            n11=np.array([item.n11 for item in types], dtype=np.int32),
            n0=np.array([item.n0 for item in types], dtype=np.int32),
            n1=np.array([item.n1 for item in types], dtype=np.int32),
            log2_counts=np.array([item.log2_count for item in types], dtype=float),
        )

    def index_of_sequence(self, sequence: Sequence[int]) -> int:
        return self.key_to_index[transition_counts(sequence)]

    def markov_scores(self, p01: float, p10: float, pi1: float | None = None) -> np.ndarray:
        eps = 1e-15
        p01 = min(max(float(p01), eps), 1.0 - eps)
        p10 = min(max(float(p10), eps), 1.0 - eps)
        if pi1 is None:
            pi1 = p01 / (p01 + p10)
        pi1 = min(max(float(pi1), eps), 1.0 - eps)
        scores = np.where(self.starts == 1, math.log2(pi1), math.log2(1.0 - pi1)).astype(float)
        scores += self.n00 * math.log2(1.0 - p01)
        scores += self.n01 * math.log2(p01)
        scores += self.n10 * math.log2(p10)
        scores += self.n11 * math.log2(1.0 - p10)
        return scores

    def kt_order1_scores(self, prior_counts: Sequence[float] | None = None) -> np.ndarray:
        if prior_counts is None:
            a00 = a01 = a10 = a11 = 0.5
        else:
            a00, a01, a10, a11 = (float(v) for v in prior_counts)
        # Fixed uniform initial symbol. Beta-binomial mixture independently per previous-state context.
        def log_beta_vec(x: np.ndarray, y: np.ndarray, ax: float, ay: float) -> np.ndarray:
            baseline = (math.lgamma(ax) + math.lgamma(ay) - math.lgamma(ax + ay)) / math.log(2.0)
            return (gammaln(x + ax) + gammaln(y + ay) - gammaln(x + y + ax + ay)) / math.log(2.0) - baseline
        return -1.0 + log_beta_vec(self.n00, self.n01, a00, a01) + log_beta_vec(self.n10, self.n11, a10, a11)

    def kt_order0_scores(self) -> np.ndarray:
        return (gammaln(self.n0 + 0.5) + gammaln(self.n1 + 0.5) - gammaln(self.n0 + self.n1 + 1.0) - 2.0 * math.lgamma(0.5)) / math.log(2.0)

    def empirical_markov_scores(self) -> np.ndarray:
        scores = np.full(len(self.types), -1.0, dtype=float)
        for count_stay, count_switch in ((self.n00, self.n01), (self.n11, self.n10)):
            total = count_stay + count_switch
            mask = total > 0
            s = np.zeros_like(scores)
            stay_mask = mask & (count_stay > 0)
            switch_mask = mask & (count_switch > 0)
            s[stay_mask] += count_stay[stay_mask] * np.log2(count_stay[stay_mask] / total[stay_mask])
            s[switch_mask] += count_switch[switch_mask] * np.log2(count_switch[switch_mask] / total[switch_mask])
            scores += s
        return scores

    def level_set_log2_ranks(self, scores: np.ndarray, tolerance: float = 1e-12) -> np.ndarray:
        scores = np.asarray(scores, dtype=float)
        if scores.shape != self.log2_counts.shape:
            raise ValueError("Score vector shape mismatch")
        order = np.argsort(-scores, kind="mergesort")
        result = np.empty_like(scores)
        cursor = 0
        cumulative_log = float("-inf")
        while cursor < len(order):
            start = cursor
            reference = scores[order[cursor]]
            while cursor < len(order) and abs(scores[order[cursor]] - reference) <= tolerance:
                cursor += 1
            group = order[start:cursor]
            cumulative_log = logsumexp2(np.concatenate(([cumulative_log], self.log2_counts[group])))
            result[group] = cumulative_log
        return result

    def rank_for_score_vector(self, scores: np.ndarray, target_index: int) -> float:
        threshold = float(scores[target_index])
        mask = scores >= threshold - 1e-12
        return logsumexp2(self.log2_counts[mask])

    def rank_under_fitted_params(self, target_index: int, pseudocount: float = 0.5) -> float:
        t = self.types[target_index]
        p01 = (t.n01 + pseudocount) / (t.n00 + t.n01 + 2.0 * pseudocount)
        p10 = (t.n10 + pseudocount) / (t.n10 + t.n11 + 2.0 * pseudocount)
        scores = self.markov_scores(p01, p10)
        return self.rank_for_score_vector(scores, target_index)
