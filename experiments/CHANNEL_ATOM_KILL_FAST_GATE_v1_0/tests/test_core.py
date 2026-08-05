from __future__ import annotations

import itertools
import math

import numpy as np

from atom_gate.channels import bac, bsc
from atom_gate.codes import BinaryLinearCode
from atom_gate.decoder import (
    direct_ml_from_codeword_array,
    one_shot_residual_decode,
    strongest_binary_exact_reference,
)
from atom_gate.enumeration import ProductAtomEnumerator
from atom_gate.metrics import channel_capacity_blahut_arimoto
from atom_gate.models import DeterministicMap, Representation
from atom_gate.representations import bac_representations, independent_row_coupling
from atom_gate.utils import binary_entropy


def test_bsc_capacity_matches_closed_form() -> None:
    for p in (0.01, 0.1, 0.2, 0.4):
        capacity, px, _ = channel_capacity_blahut_arimoto(bsc(p).matrix)
        assert math.isclose(capacity, 1.0 - binary_entropy(p), abs_tol=2e-10)
        assert np.allclose(px, [0.5, 0.5], atol=2e-8)


def test_duplicate_reduction_preserves_channel() -> None:
    identity = DeterministicMap((0, 1))
    flip = DeterministicMap((1, 0))
    rep = Representation("split", [identity, identity, flip], np.array([0.2, 0.5, 0.3]))
    reduced = rep.reduced()
    assert reduced.support_size == 2
    assert np.allclose(rep.induced_channel(2), reduced.induced_channel(2))


def test_bac_family_is_exact_and_one_shot_ml() -> None:
    spec = bac(0.17, 0.31)
    for rep in bac_representations(0.17, 0.31, grid_points=9):
        rep.verify(spec.matrix)
        for y in range(2):
            result = one_shot_residual_decode(spec.matrix, rep.sorted_by_weight(), y)
            assert result.exact


def test_product_enumerator_is_complete_and_ordered() -> None:
    probabilities = np.array([0.55, 0.30, 0.15])
    n = 3
    items = list(ProductAtomEnumerator(probabilities, n))
    assert len(items) == len(probabilities) ** n
    observed = [item.probability for item in items]
    assert all(a + 1e-14 >= b for a, b in zip(observed, observed[1:]))
    assert math.isclose(sum(observed), 1.0, abs_tol=2e-13)
    assert len({item.atom_indices for item in items}) == len(items)


def test_binary_code_parity_check_and_fiber_oracle() -> None:
    rng = np.random.default_rng(7)
    code = BinaryLinearCode.random_systematic(8, 5, rng)
    assert code.parity_check.shape == (3, 8)
    assert not np.any((code.generator @ code.parity_check.T) % 2)
    for fixed_mask in (0, 0b11, 0b10110100, 0xFF):
        for fixed_value in (0, 1, 0b10010000, 0xFF):
            fixed_value &= fixed_mask
            solution = code.fiber(fixed_mask, fixed_value)
            brute = np.array(
                [
                    i
                    for i, word in enumerate(code.codewords_int)
                    if ((int(word) ^ fixed_value) & fixed_mask) == 0
                ],
                dtype=np.int64,
            )
            assert np.array_equal(np.sort(solution.message_indices), brute)


def test_syndrome_trellis_matches_direct_ml() -> None:
    rng = np.random.default_rng(9)
    code = BinaryLinearCode.random_systematic(10, 7, rng)
    channels = [bsc(0.12).matrix, bac(0.15, 0.33).matrix]
    for channel in channels:
        for _ in range(40):
            received = tuple(int(v) for v in rng.integers(0, channel.shape[1], size=code.n))
            reference = strongest_binary_exact_reference(channel, code, received)
            decision, ties, _, _ = direct_ml_from_codeword_array(
                channel, code.codewords_array, received
            )
            assert decision in ties
            assert int(reference["decision"]) in ties
            assert str(reference["selected_name"]) in {
                "direct_codeword_ml",
                "syndrome_trellis_ml",
            }


def test_independent_coupling_matches_channel() -> None:
    channel = np.array([[0.65, 0.25, 0.10], [0.15, 0.55, 0.30]])
    rep = independent_row_coupling(channel)
    rep.verify(channel)
