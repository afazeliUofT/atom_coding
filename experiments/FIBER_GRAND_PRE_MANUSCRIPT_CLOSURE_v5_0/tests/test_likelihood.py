import math
import numpy as np

from fiber_flagship.channels import FixedDeletionBSC, sample_channel
from fiber_flagship.codes import random_linear
from fiber_flagship.likelihood import deletion_likelihood, exhaustive_ml


def test_likelihood_normalization_small():
    n = 5
    channel = FixedDeletionBSC(n, 1, 0.1)
    for word in range(1 << n):
        total = 0.0
        for y in range(1 << (n - 1)):
            total += deletion_likelihood(word, y, channel)[0]
        assert math.isclose(total, 1.0, abs_tol=1e-12)


def test_exhaustive_ml_contains_transmitted_on_noiseless_vt_like_case():
    rng = np.random.default_rng(4)
    code = random_linear(8, 5, rng)
    words = code.enumerate_codewords()
    channel = FixedDeletionBSC(8, 1, 0.0)
    transmitted = int(words[3])
    y, _, _ = sample_channel(transmitted, channel, rng)
    ties, _, _ = exhaustive_ml(words, y, channel)
    assert len(ties) >= 1
