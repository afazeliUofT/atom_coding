from __future__ import annotations

import heapq
import math
import time
from dataclasses import dataclass

import numpy as np
import psutil

from .channels import FixedDeletionBSC
from .codes import LinearOracle
from .likelihood import deletion_likelihood
from .utils import binary_tuple
from .work import Work


@dataclass(frozen=True)
class PrefixOutcome:
    decision_word: int | None
    tie_words: tuple[int, ...]
    certified: bool
    incumbent_score: float
    frontier_bound: float
    work: Work


@dataclass
class Node:
    bound: float
    length: int
    prefix: int
    mismatches: np.ndarray


def _extend(parent: np.ndarray, position: int, bit: int, y: tuple[int, ...]) -> tuple[np.ndarray, int]:
    n = len(parent)
    child = parent.copy()
    terms = 0
    for deletion_position in range(n):
        if position == deletion_position:
            continue
        output_index = position if position < deletion_position else position - 1
        if 0 <= output_index < len(y):
            child[deletion_position] += int(bit != y[output_index])
            terms += 1
    return child, terms


def _mass_table(channel: FixedDeletionBSC) -> np.ndarray:
    if channel.p == 0.0:
        table = np.zeros(channel.m + 1, dtype=float)
        table[0] = 1.0
        return table
    return np.asarray([
        (channel.p**d) * ((1.0 - channel.p) ** (channel.m - d))
        for d in range(channel.m + 1)
    ], dtype=float)


def prefix_aggregate_astar(
    received: int,
    channel: FixedDeletionBSC,
    code: LinearOracle,
    max_nodes: int,
    tolerance: float = 1e-14,
) -> PrefixOutcome:
    if channel.deletions != 1:
        raise ValueError("prefix baseline supports one deletion")
    start = time.perf_counter()
    y = binary_tuple(received, channel.m)
    masses = _mass_table(channel)

    def bound(mismatches: np.ndarray) -> float:
        clipped = np.minimum(mismatches, channel.m)
        return float(np.mean(masses[clipped]))

    root_mismatch = np.zeros(channel.n, dtype=np.int16)
    root = Node(bound(root_mismatch), 0, 0, root_mismatch)
    heap: list[tuple[float, int, int, Node]] = [(-root.bound, 0, 0, root)]
    serial = 1
    work = Work(heap_pushes=1, peak_frontier=1)
    best = -1.0
    best_words: set[int] = set()
    certified = False
    frontier_bound = root.bound

    while heap and work.trellis_nodes < max_nodes:
        neg, _, _, node = heapq.heappop(heap)
        work.heap_pops += 1
        work.trellis_nodes += 1
        frontier_bound = -float(neg)
        if best_words and math.nextafter(best, -math.inf) > math.nextafter(frontier_bound, math.inf) + tolerance:
            certified = True
            break
        if node.length == channel.n:
            work.trellis_terminals += 1
            valid, bitops = code.is_codeword(node.prefix)
            work.membership_queries += 1
            work.syndrome_bitops += bitops
            if valid:
                score, ops = deletion_likelihood(node.prefix, received, channel)
                work.exact_score_calls += 1
                work.likelihood_ops += ops
                if score > best + tolerance:
                    best = score
                    best_words = {node.prefix}
                elif abs(score - best) <= tolerance:
                    best_words.add(node.prefix)
            continue
        position = node.length
        for bit in (0, 1):
            prefix = node.prefix | (bit << position)
            feasible, ops = code.prefix_feasible_systematic(prefix, position + 1)
            work.syndrome_bitops += ops
            if not feasible:
                continue
            mismatches, terms = _extend(node.mismatches, position, bit, y)
            work.trellis_dp_updates += terms
            bnd = bound(mismatches)
            work.bound_checks += 1
            if not best_words or bnd >= best - tolerance:
                child = Node(bnd, position + 1, prefix, mismatches)
                heapq.heappush(heap, (-bnd, -(position + 1), serial, child))
                serial += 1
                work.heap_pushes += 1
        work.peak_frontier = max(work.peak_frontier, len(heap))

    if not certified and best_words:
        frontier_bound = -heap[0][0] if heap else 0.0
        certified = math.nextafter(best, -math.inf) > math.nextafter(frontier_bound, math.inf) + tolerance
    work.wall_seconds = time.perf_counter() - start
    work.peak_rss_bytes = int(psutil.Process().memory_info().rss)
    return PrefixOutcome(
        decision_word=min(best_words) if best_words else None,
        tie_words=tuple(sorted(best_words)),
        certified=certified,
        incumbent_score=max(0.0, best),
        frontier_bound=max(0.0, frontier_bound),
        work=work,
    )
