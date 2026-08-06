from __future__ import annotations

import math

from fiber_flagship.candidate_theory import (
    ambiguity_degree,
    candidate_volume_bounds,
    cumulative_hamming_ball,
    extra_shells,
)


def test_candidate_volume_bounds_basic() -> None:
    bounds = candidate_volume_bounds(8, 1, 2)
    assert bounds.lower == 2 * cumulative_hamming_ball(7, 2)
    assert bounds.upper == 8 * bounds.lower


def test_ambiguity_degree() -> None:
    assert ambiguity_degree(8, 1) == 8
    assert ambiguity_degree(8, 2) == math.comb(8, 2)


def test_extra_shells_strict_shift() -> None:
    n, t, p = 32, 1, 0.05
    ell = extra_shells(n, t, p)
    a = (1.0 - p) / p
    assert a ** (ell + 1) > math.comb(n, t)
