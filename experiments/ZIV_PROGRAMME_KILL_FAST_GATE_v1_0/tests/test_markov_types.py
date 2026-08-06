import itertools
import math
from collections import Counter

from ziv_gate.markov_types import MarkovTypeAtlas, enumerate_binary_markov_types, transition_counts


def test_type_counts_exact():
    for n in range(1, 10):
        exact = Counter(transition_counts(bits) for bits in itertools.product((0, 1), repeat=n))
        types = enumerate_binary_markov_types(n)
        assert len(types) == len(exact)
        for item in types:
            assert math.isclose(item.log2_count, math.log2(exact[item.key]), abs_tol=1e-10)


def test_level_set_ranks_bounded():
    atlas = MarkovTypeAtlas.build(32)
    for scores in (
        atlas.kt_order0_scores(),
        atlas.kt_order1_scores(),
        atlas.empirical_markov_scores(),
        atlas.markov_scores(0.05, 0.2),
    ):
        ranks = atlas.level_set_log2_ranks(scores)
        assert ranks.min() >= 0
        assert ranks.max() <= 32 + 1e-8
