from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass

import psutil

from .codes import VTOracle
from .utils import insert_hidden_bits
from .work import Work


@dataclass(frozen=True)
class VTOutcome:
    decision_word: int | None
    tie_words: tuple[int, ...]
    score: int
    work: Work


def vt_direct_one_deletion(received: int, code: VTOracle) -> VTOutcome:
    start = time.perf_counter()
    counts: Counter[int] = Counter()
    work = Work()
    for position in range(code.n):
        for bit in (0, 1):
            candidate = insert_hidden_bits(received, code.n - 1, (position,), bit)
            valid, ops = code.is_codeword(candidate)
            work.vt_candidate_checks += 1
            work.membership_queries += 1
            work.syndrome_bitops += ops
            if valid:
                counts[candidate] += 1
    best = max(counts.values(), default=0)
    ties = tuple(sorted(word for word, count in counts.items() if count == best))
    work.wall_seconds = time.perf_counter() - start
    work.peak_rss_bytes = int(psutil.Process().memory_info().rss)
    return VTOutcome(min(ties) if ties else None, ties, best, work)
