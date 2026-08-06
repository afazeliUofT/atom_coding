import math

from fiber_v2.channels import FixedDeletionBSC
from fiber_v2.shell_theory import certificate_inequality_holds, shell_certificate_bound


def test_shell_inequality_all_small():
    for n in (8, 12, 20):
        for t in (1, 2):
            for p in (0.0, 0.02, 0.05, 0.1, 0.2):
                channel = FixedDeletionBSC(n, t, p)
                for e in range(channel.m + 1):
                    assert certificate_inequality_holds(channel, e)


def test_noiseless_bound_is_polynomial_count():
    channel = FixedDeletionBSC(20, 2, 0.0)
    bound = shell_certificate_bound(channel, 0)
    assert bound.history_upper_bound == 4 * math.comb(20, 2)


def test_bound_contains_hidden_assignments_and_streams():
    channel = FixedDeletionBSC(16, 1, 0.05)
    bound = shell_certificate_bound(channel, 1)
    assert bound.history_upper_bound >= 2 * 16
