from __future__ import annotations

import heapq
import math
import time
from dataclasses import dataclass

import numpy as np
import psutil

from .channels import FixedDeletionBSC
from .codes import LinearOracle
from .likelihood import component_probability, deletion_likelihood
from .utils import binary_tuple
from .work import Work


@dataclass(frozen=True)
class TrellisOutcome:
    decision_word: int | None
    tie_words: tuple[int, ...]
    certified: bool
    incumbent_score: float
    residual_bound: float
    work: Work

    def to_dict(self) -> dict:
        return {
            "decision_word": self.decision_word,
            "tie_words": list(self.tie_words),
            "certified": self.certified,
            "incumbent_score": self.incumbent_score,
            "residual_bound": self.residual_bound,
            "work": self.work.to_dict(),
        }


class AlignmentCodewordStream:
    """A* enumerator of codewords in nondecreasing mismatch cost for one alignment."""

    def __init__(self, received: int, deleted_position: int, channel: FixedDeletionBSC, code: LinearOracle, work: Work) -> None:
        if channel.deletions != 1:
            raise ValueError("syndrome-trellis baseline currently supports one deletion")
        self.n = channel.n
        self.r = code.redundancy
        self.columns = code.syndrome_columns
        y = binary_tuple(received, channel.m)
        target: list[int | None] = []
        y_index = 0
        for position in range(channel.n):
            if position == deleted_position:
                target.append(None)
            else:
                target.append(y[y_index])
                y_index += 1
        self.target = target
        states = 1 << self.r
        inf = self.n + 1
        dp = np.full((self.n + 1, states), inf, dtype=np.int16)
        dp[self.n, 0] = 0
        updates = 0
        for pos in range(self.n - 1, -1, -1):
            column = self.columns[pos]
            for required in range(states):
                c0 = 0 if target[pos] is None or target[pos] == 0 else 1
                c1 = 0 if target[pos] is None or target[pos] == 1 else 1
                v0 = c0 + int(dp[pos + 1, required])
                v1 = c1 + int(dp[pos + 1, required ^ column])
                dp[pos, required] = min(v0, v1)
                updates += 2
        self.dp = dp
        work.trellis_dp_updates += updates
        self.heap: list[tuple[int, int, int, int, int]] = []
        self.serial = 0
        root_h = int(dp[0, 0])
        heapq.heappush(self.heap, (root_h, 0, self.serial, 0, 0))  # f,g,serial,pos,packed(syndrome<<n|word)
        self.serial += 1
        work.heap_pushes += 1

    def next(self, work: Work) -> tuple[int, int] | None:
        nmask = (1 << self.n) - 1
        while self.heap:
            f, g, _, pos, packed = heapq.heappop(self.heap)
            work.heap_pops += 1
            work.trellis_nodes += 1
            syndrome = packed >> self.n
            word = packed & nmask
            if pos == self.n:
                if syndrome == 0:
                    work.trellis_terminals += 1
                    return word, g
                continue
            target = self.target[pos]
            column = self.columns[pos]
            for bit in (0, 1):
                next_syndrome = syndrome ^ (column if bit else 0)
                cost = 0 if target is None or target == bit else 1
                ng = g + cost
                nh = int(self.dp[pos + 1, next_syndrome])
                if nh > self.n:
                    continue
                next_word = word | (bit << pos)
                next_packed = (next_syndrome << self.n) | next_word
                heapq.heappush(self.heap, (ng + nh, ng, self.serial, pos + 1, next_packed))
                self.serial += 1
                work.heap_pushes += 1
            work.peak_frontier = max(work.peak_frontier, len(self.heap))
        return None


def syndrome_trellis_aggregate_decode(
    received: int,
    channel: FixedDeletionBSC,
    code: LinearOracle,
    max_terminals: int,
    tolerance: float = 1e-13,
) -> TrellisOutcome:
    if channel.deletions != 1:
        raise ValueError("baseline supports exactly one deletion")
    start = time.perf_counter()
    work = Work()
    streams = [AlignmentCodewordStream(received, j, channel, code, work) for j in range(channel.n)]
    heap: list[tuple[float, int, int, int, int]] = []  # -prob, stream, serial, word, cost
    eta = [0.0] * channel.n
    serial = 0
    for j, stream in enumerate(streams):
        item = stream.next(work)
        if item is not None:
            word, cost = item
            prob = component_probability(cost, channel.m, channel.p, channel.n)
            eta[j] = prob
            heapq.heappush(heap, (-prob, j, serial, word, cost))
            serial += 1
            work.heap_pushes += 1
    residual = sum(eta)
    seen: set[int] = set()
    best = -1.0
    best_words: set[int] = set()
    certified = False

    while heap and work.trellis_terminals <= max_terminals:
        _, j, _, word, cost = heapq.heappop(heap)
        work.heap_pops += 1
        old_eta = eta[j]
        next_item = streams[j].next(work)
        if next_item is None:
            eta[j] = 0.0
        else:
            next_word, next_cost = next_item
            prob = component_probability(next_cost, channel.m, channel.p, channel.n)
            eta[j] = prob
            heapq.heappush(heap, (-prob, j, serial, next_word, next_cost))
            serial += 1
            work.heap_pushes += 1
        residual = max(0.0, residual + eta[j] - old_eta)
        if word in seen:
            work.duplicates += 1
        else:
            seen.add(word)
            work.distinct_candidates += 1
            score, ops = deletion_likelihood(word, received, channel)
            work.exact_score_calls += 1
            work.likelihood_ops += ops
            if score > best + tolerance:
                best = score
                best_words = {word}
            elif abs(score - best) <= tolerance:
                best_words.add(word)
        work.bound_checks += 1
        if best_words and best > residual + tolerance:
            certified = True
            break

    work.wall_seconds = time.perf_counter() - start
    work.peak_rss_bytes = int(psutil.Process().memory_info().rss)
    return TrellisOutcome(
        decision_word=min(best_words) if best_words else None,
        tie_words=tuple(sorted(best_words)),
        certified=certified,
        incumbent_score=max(0.0, best),
        residual_bound=residual,
        work=work,
    )
