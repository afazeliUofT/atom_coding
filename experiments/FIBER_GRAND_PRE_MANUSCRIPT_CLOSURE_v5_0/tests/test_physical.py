import numpy as np

from fiber_flagship.physical_model import simulate_timing_slip_bits


def test_timing_slip_model_has_one_deletion():
    rng = np.random.default_rng(11)
    for _ in range(50):
        bits = tuple(int(v) for v in rng.integers(0, 2, size=32))
        received, deleted, errors = simulate_timing_slip_bits(bits, 7.0, 0.06, rng)
        assert 0 <= deleted < 32
        assert len(errors) == 31
        assert 0 <= received < (1 << 31)
