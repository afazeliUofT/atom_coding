from __future__ import annotations

import numpy as np

from fiber_gate.baselines import exhaustive_deletion_ml, exhaustive_insertion_ml
from fiber_gate.channels import DeletionChannel, InsertionChannel
from fiber_gate.codes import SystematicLinearCode
from fiber_gate.history_decoder import deletion_history_model, history_fiber_decode, insertion_history_model
from fiber_gate.prefix_decoder import prefix_fiber_decode


def _code() -> SystematicLinearCode:
    return SystematicLinearCode.random_systematic(9, 6, np.random.default_rng(4), "TEST")


def test_history_one_deletion_is_exact() -> None:
    code = _code()
    channel = DeletionChannel(9, 1, 0.08)
    for received in range(1 << 8):
        reference = exhaustive_deletion_ml(code, received, channel)
        result = history_fiber_decode(deletion_history_model(received, channel), code, reference.tie_set, reference.decision, reference.work, 100000)
        assert result.exact and result.certified and not result.fallback_used


def test_prefix_one_deletion_is_exact() -> None:
    code = _code()
    channel = DeletionChannel(9, 1, 0.08)
    for received in range(0, 1 << 8, 7):
        reference = exhaustive_deletion_ml(code, received, channel)
        result = prefix_fiber_decode(received, channel, code, reference.tie_set, reference.decision, reference.work, 100000, True)
        assert result.exact and result.certified and not result.fallback_used


def test_history_two_deletion_is_exact() -> None:
    code = _code()
    channel = DeletionChannel(9, 2, 0.05)
    for received in range(1 << 7):
        reference = exhaustive_deletion_ml(code, received, channel)
        result = history_fiber_decode(deletion_history_model(received, channel), code, reference.tie_set, reference.decision, reference.work, 250000)
        assert result.exact and result.certified and not result.fallback_used


def test_history_insertion_is_exact() -> None:
    code = _code()
    channel = InsertionChannel(9, 0.05)
    for received in range(0, 1 << 10, 11):
        reference = exhaustive_insertion_ml(code, received, channel)
        result = history_fiber_decode(insertion_history_model(received, channel), code, reference.tie_set, reference.decision, reference.work, 250000)
        assert result.exact and result.certified and not result.fallback_used
