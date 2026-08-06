import numpy as np

from fiber_v2.channels import FixedDeletionBSC, sample_channel
from fiber_v2.codes import VTOracle, random_linear
from fiber_v2.history_decoder import history_decode
from fiber_v2.likelihood import exhaustive_ml
from fiber_v2.syndrome_trellis import syndrome_trellis_aggregate_decode
from fiber_v2.prefix_astar import prefix_aggregate_astar
from fiber_v2.vt_baseline import vt_direct_one_deletion


def test_history_and_trellis_match_exhaustive():
    rng = np.random.default_rng(5)
    code = random_linear(8, 5, rng)
    words = code.enumerate_codewords()
    channel = FixedDeletionBSC(8, 1, 0.05)
    for _ in range(10):
        x = code.sample_codeword(rng)
        y, _, _ = sample_channel(x, channel, rng)
        ties, _, _ = exhaustive_ml(words, y, channel)
        tie_words = {int(words[i]) for i in ties}
        f = history_decode(y, channel, code, 100000)
        t = syndrome_trellis_aggregate_decode(y, channel, code, 100000)
        a = prefix_aggregate_astar(y, channel, code, 100000)
        assert f.certified and t.certified and a.certified
        assert f.decision_word in tie_words
        assert t.decision_word in tie_words
        assert a.decision_word in tie_words


def test_two_deletion_history_matches_exhaustive():
    rng = np.random.default_rng(6)
    code = random_linear(8, 5, rng)
    words = code.enumerate_codewords()
    channel = FixedDeletionBSC(8, 2, 0.05)
    for _ in range(8):
        x = code.sample_codeword(rng)
        y, _, _ = sample_channel(x, channel, rng)
        ties, _, _ = exhaustive_ml(words, y, channel)
        tie_words = {int(words[i]) for i in ties}
        f = history_decode(y, channel, code, 200000)
        assert f.certified
        assert f.decision_word in tie_words


def test_vt_specialized_ties_match():
    rng = np.random.default_rng(7)
    code = VTOracle(11, 0)
    channel = FixedDeletionBSC(11, 1, 0.0)
    for _ in range(10):
        x = code.sample_codeword(rng)
        y, _, _ = sample_channel(x, channel, rng)
        f = history_decode(y, channel, code, 10000)
        v = vt_direct_one_deletion(y, code)
        assert f.certified
        assert set(f.tie_words) == set(v.tie_words)
