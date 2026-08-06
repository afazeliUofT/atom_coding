from __future__ import annotations

import numpy as np

from fiber_gate.channels import DeletionChannel, InsertionChannel
from fiber_gate.likelihood import (
    deletion_likelihood_dp,
    deletion_likelihood_vectorized,
    insertion_likelihood_vectorized,
    one_deletion_likelihood,
    one_insertion_likelihood,
)
from fiber_gate.utils import int_array


def test_one_deletion_recurrence_matches_dp() -> None:
    channel = DeletionChannel(7, 1, 0.13)
    for word in range(1 << 7):
        for received in range(1 << 6):
            assert abs(one_deletion_likelihood(word, received, channel) - deletion_likelihood_dp(word, received, channel)) < 1e-12


def test_deletion_vectorized_matches_scalar() -> None:
    channel = DeletionChannel(8, 2, 0.08)
    words = np.arange(1 << 8, dtype=np.uint64)
    array = int_array(words, 8)
    scores, _ = deletion_likelihood_vectorized(array, 37, channel)
    expected = np.array([deletion_likelihood_dp(int(word), 37, channel) for word in words])
    assert np.max(np.abs(scores - expected)) < 1e-12


def test_insertion_vectorized_matches_scalar() -> None:
    channel = InsertionChannel(7, 0.07)
    words = np.arange(1 << 7, dtype=np.uint64)
    array = int_array(words, 7)
    scores, _ = insertion_likelihood_vectorized(array, 149, channel)
    expected = np.array([one_insertion_likelihood(int(word), 149, channel) for word in words])
    assert np.max(np.abs(scores - expected)) < 1e-12
