from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class MarkovRegime:
    name: str
    p01: float
    p10: float
    structured: bool
    description: str

    @property
    def stationary_one(self) -> float:
        return self.p01 / (self.p01 + self.p10)


DEFAULT_REGIMES = (
    MarkovRegime("BURSTY_RARE", 0.010, 0.200, True, "rare bursts with short recovery"),
    MarkovRegime("STICKY_BALANCED", 0.030, 0.030, True, "long alternating regimes with marginal one-half"),
    MarkovRegime("MODERATE_MEMORY", 0.080, 0.400, True, "moderate low-weight Markov errors"),
    MarkovRegime("NEAR_IID_CONTROL", 0.080, 0.920, False, "iid Bernoulli(0.08) represented as a Markov chain"),
)


def sample_markov(n: int, p01: float, p10: float, rng: np.random.Generator, initial: int | None = None) -> tuple[int, ...]:
    if n <= 0:
        raise ValueError("n must be positive")
    if initial is None:
        pi1 = p01 / (p01 + p10)
        bit = int(rng.random() < pi1)
    else:
        bit = int(initial)
    output = [bit]
    for _ in range(n - 1):
        if bit == 0:
            bit = int(rng.random() < p01)
        else:
            bit = 0 if rng.random() < p10 else 1
        output.append(bit)
    return tuple(output)


def transition_counts_array(sequence: Sequence[int]) -> tuple[int, int, int, int]:
    n00 = n01 = n10 = n11 = 0
    for a, b in zip(sequence, sequence[1:]):
        if a == 0 and b == 0:
            n00 += 1
        elif a == 0 and b == 1:
            n01 += 1
        elif a == 1 and b == 0:
            n10 += 1
        else:
            n11 += 1
    return n00, n01, n10, n11


def fit_markov(sequence: Sequence[int], pseudocount: float = 0.5) -> tuple[float, float]:
    n00, n01, n10, n11 = transition_counts_array(sequence)
    p01 = (n01 + pseudocount) / (n00 + n01 + 2.0 * pseudocount)
    p10 = (n10 + pseudocount) / (n10 + n11 + 2.0 * pseudocount)
    return float(p01), float(p10)
