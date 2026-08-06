import math

from fiber_flagship.moment_theory import exact_moment_rates, exponential_tail_quantile_exponent
from fiber_flagship.phase_diagram import threshold_for_target
from fiber_flagship.utils import binary_renyi, h2


def test_moment_sandwich_and_convergence_direction():
    for p in (0.01, 0.05):
        for rho in (0.5, 1.0, 2.0):
            small = exact_moment_rates(64, 1, p, rho)
            large = exact_moment_rates(512, 1, p, rho)
            assert small.reveal_lower <= small.reveal_upper <= small.certificate_upper + 1e-12
            assert large.reveal_lower <= large.reveal_upper <= large.certificate_upper + 1e-12
            assert abs(large.reveal_upper - large.theory) <= abs(small.reveal_upper - small.theory) + 0.05


def test_phase_thresholds():
    for rate in (0.625, 0.75, 0.875):
        target = 1.0 - rate
        typical = threshold_for_target(target, "typical")
        mean = threshold_for_target(target, "mean")
        assert mean < typical
        assert math.isclose(h2(typical), target, abs_tol=1e-10)
        assert math.isclose(binary_renyi(mean, 0.5), target, abs_tol=1e-10)


def test_exponential_tail_quantile_monotone():
    values = [exponential_tail_quantile_exponent(0.02, beta) for beta in (0.0, 0.01, 0.05, 0.1)]
    assert all(b >= a for a, b in zip(values, values[1:]))
