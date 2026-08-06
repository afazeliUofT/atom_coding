from __future__ import annotations

import heapq
import itertools
import time
from dataclasses import dataclass
from typing import Iterator, Sequence

import psutil

from .channels import FixedDeletionBSC
from .codes import CodeOracle
from .likelihood import component_probability, deletion_likelihood
from .utils import insert_hidden_bits, masks_of_weight
from .work import Work


@dataclass(frozen=True)
class DecodeOutcome:
    decision_word: int | None
    tie_words: tuple[int, ...]
    certified: bool
    incumbent_score: float
    residual_bound: float
    work: Work
    shell_at_stop: int

    def to_dict(self) -> dict:
        return {
            "decision_word": self.decision_word,
            "tie_words": list(self.tie_words),
            "certified": self.certified,
            "incumbent_score": self.incumbent_score,
            "residual_bound": self.residual_bound,
            "shell_at_stop": self.shell_at_stop,
            "work": self.work.to_dict(),
        }


@dataclass(frozen=True)
class _Item:
    probability: float
    error_mask: int
    hidden: int
    shell: int


class _ShellStream:
    def __init__(self, m: int, t: int, p: float, streams: int) -> None:
        self.m = m
        self.t = t
        self.p = p
        self.streams = streams
        self.iterator = self._items()

    def _items(self) -> Iterator[_Item]:
        weights = [0] if self.p == 0.0 else range(self.m + 1)
        for weight in weights:
            probability = component_probability(weight, self.m, self.p, self.streams)
            if probability <= 0.0:
                continue
            for error_mask in masks_of_weight(self.m, weight):
                for hidden in range(1 << self.t):
                    yield _Item(probability, error_mask, hidden, weight)

    def next(self) -> _Item | None:
        return next(self.iterator, None)


def history_decode(
    received: int,
    channel: FixedDeletionBSC,
    code: CodeOracle,
    max_histories: int,
    tolerance: float = 1e-13,
) -> DecodeOutcome:
    start = time.perf_counter()
    positions = channel.deletion_subsets
    streams = [_ShellStream(channel.m, channel.deletions, channel.p, len(positions)) for _ in positions]
    heap: list[tuple[float, int, int, _Item]] = []
    eta = [0.0] * len(streams)
    serial = 0
    work = Work()
    for idx, stream in enumerate(streams):
        item = stream.next()
        if item is not None:
            eta[idx] = item.probability
            heapq.heappush(heap, (-item.probability, idx, serial, item))
            serial += 1
            work.heap_pushes += 1
    work.peak_frontier = len(heap)
    residual = sum(eta)
    seen: set[int] = set()
    scores: dict[int, float] = {}
    best = -1.0
    best_words: set[int] = set()
    certified = False
    shell_at_stop = 0

    while heap and work.histories < max_histories:
        _, stream_idx, _, item = heapq.heappop(heap)
        work.heap_pops += 1
        old_eta = eta[stream_idx]
        next_item = streams[stream_idx].next()
        if next_item is None:
            eta[stream_idx] = 0.0
        else:
            eta[stream_idx] = next_item.probability
            heapq.heappush(heap, (-next_item.probability, stream_idx, serial, next_item))
            serial += 1
            work.heap_pushes += 1
        residual = max(0.0, residual + eta[stream_idx] - old_eta)
        work.histories += 1
        work.candidate_build_ops += channel.n
        shell_at_stop = max(shell_at_stop, item.shell)
        base = int(received) ^ int(item.error_mask)
        candidate = insert_hidden_bits(base, channel.m, positions[stream_idx], item.hidden)
        if candidate in seen:
            work.duplicates += 1
        else:
            seen.add(candidate)
            work.distinct_candidates += 1
            work.peak_seen = max(work.peak_seen, len(seen))
            valid, bitops = code.is_codeword(candidate)
            work.membership_queries += 1
            work.syndrome_bitops += bitops
            if valid:
                score, ops = deletion_likelihood(candidate, received, channel)
                scores[candidate] = score
                work.exact_score_calls += 1
                work.likelihood_ops += ops
                if score > best + tolerance:
                    best = score
                    best_words = {candidate}
                elif abs(score - best) <= tolerance:
                    best_words.add(candidate)
        work.bound_checks += 1
        work.peak_frontier = max(work.peak_frontier, len(heap))
        if best_words and best > residual + tolerance:
            certified = True
            break

    work.wall_seconds = time.perf_counter() - start
    work.peak_rss_bytes = int(psutil.Process().memory_info().rss)
    decision = min(best_words) if best_words else None
    return DecodeOutcome(
        decision_word=decision,
        tie_words=tuple(sorted(best_words)),
        certified=certified,
        incumbent_score=max(0.0, best),
        residual_bound=residual,
        work=work,
        shell_at_stop=shell_at_stop,
    )
