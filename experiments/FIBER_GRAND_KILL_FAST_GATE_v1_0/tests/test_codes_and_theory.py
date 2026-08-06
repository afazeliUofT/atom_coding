from __future__ import annotations

import numpy as np

from fiber_gate.codes import SystematicLinearCode, VTCode, crc_systematic_code, hamming_15_11_code
from fiber_gate.theory_audit import fiber_guesswork_identity, multiplicity_counterexample, random_code_rank_identity


def test_systematic_prefix_feasibility() -> None:
    code = SystematicLinearCode.random_systematic(12, 8, np.random.default_rng(1))
    for message in range(code.size):
        word = code.encode_message(message)
        for length in range(13):
            assert code.prefix_feasible(word & ((1 << length) - 1), length)[0]
    corrupted = code.encode_message(3) ^ (1 << 10)
    assert not code.prefix_feasible(corrupted, 11)[0]


def test_code_families_are_closed_and_distinct() -> None:
    codes = [
        crc_systematic_code(12, 8, 0b10011),
        hamming_15_11_code(),
        VTCode(11, 0),
    ]
    for code in codes:
        assert code.size == len(code.membership)
        assert len(np.unique(code.codewords_int)) == code.size
        for word in code.codewords_int[: min(64, code.size)]:
            assert code.is_codeword(int(word))


def test_fiber_guesswork_is_posterior_guesswork() -> None:
    result = fiber_guesswork_identity(max_n=6)
    assert result["pass"]
    assert result["maximum_order_disagreement"] == 0


def test_random_code_rank_identity() -> None:
    result = random_code_rank_identity(max_n=5)
    assert result["pass"]


def test_multiplicity_counterexample() -> None:
    result = multiplicity_counterexample()
    assert result["pass"]
    assert result["scores"]["000"] > result["scores"]["001"]
