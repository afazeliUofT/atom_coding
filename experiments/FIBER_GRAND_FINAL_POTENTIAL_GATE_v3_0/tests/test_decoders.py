import numpy as np

from fiber_final.channels import FixedDeletionBSC, sample_channel
from fiber_final.codes import VTOracle, random_linear
from fiber_final.history_decoder import history_decode
from fiber_final.likelihood import exhaustive_ml
from fiber_final.prefix_astar import prefix_aggregate_astar
from fiber_final.syndrome_trellis import syndrome_trellis_aggregate_decode
from fiber_final.vt_linear import vt_decode_single_deletion_linear


def test_history_trellis_prefix_complete_ties_match_exhaustive():
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
        assert set(f.tie_words) == tie_words
        assert set(t.tie_words) == tie_words
        assert set(a.tie_words) == tie_words


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
        assert set(f.tie_words) == tie_words


def test_vt_linear_decoder_matches_transmitted():
    rng = np.random.default_rng(7)
    code = VTOracle(11, 0)
    channel = FixedDeletionBSC(11, 1, 0.0)
    for _ in range(20):
        x = code.sample_codeword(rng)
        y, _, _ = sample_channel(x, channel, rng)
        f = history_decode(y, channel, code, 10000)
        v = vt_decode_single_deletion_linear(y, code)
        assert f.certified and v.valid
        assert f.decision_word == v.word == x
