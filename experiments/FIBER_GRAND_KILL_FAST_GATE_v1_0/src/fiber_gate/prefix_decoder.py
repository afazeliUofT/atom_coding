from __future__ import annotations

import heapq
import time
from dataclasses import dataclass

import numpy as np
import psutil

from .channels import DeletionChannel
from .codes import BinaryCode
from .likelihood import deletion_likelihood_dp
from .models import DecodeResult, WorkVector
from .utils import binary_tuple


@dataclass
class _PrefixNode:
    bound: float
    length: int
    prefix: int
    mismatch_vector: np.ndarray


def _initial_mismatch_vector(n: int) -> np.ndarray:
    return np.zeros(n, dtype=np.int16)


def _extend_mismatch_vector(
    parent: np.ndarray,
    position: int,
    bit: int,
    received_bits: tuple[int, ...],
) -> tuple[np.ndarray, int]:
    n = len(parent)
    child = parent.copy()
    terms = 0
    for deletion_position in range(n):
        if position == deletion_position:
            continue
        output_index = position if position < deletion_position else position - 1
        if 0 <= output_index < len(received_bits):
            child[deletion_position] += int(bit != received_bits[output_index])
            terms += 1
    return child, terms


def _bound_from_mismatches(
    mismatches: np.ndarray,
    channel: DeletionChannel,
) -> float:
    if channel.deletions != 1:
        raise ValueError("Prefix decoder currently supports exactly one deletion")
    q = (
        np.full(channel.n, 1.0 / channel.n, dtype=float)
        if channel.deletion_weights is None
        else np.asarray(channel.deletion_weights, dtype=float)
    )
    if channel.substitution_probability == 0.0:
        masses = (mismatches == 0).astype(float)
    else:
        p = channel.substitution_probability
        masses = np.power(p, mismatches) * np.power(1.0 - p, channel.n - 1 - mismatches)
    return float(np.dot(q, masses))


def prefix_fiber_decode(
    received: int,
    channel: DeletionChannel,
    code: BinaryCode,
    ml_tie_set: tuple[int, ...],
    reference_decision: int,
    reference_work: WorkVector,
    max_nodes: int,
    use_prefix_feasibility: bool,
    tolerance: float = 1e-13,
) -> DecodeResult:
    if channel.deletions != 1:
        raise ValueError("Prefix FIBER-GRAND is implemented only for one deletion")
    start = time.perf_counter()
    y = binary_tuple(received, channel.n - 1)
    initial_mismatches = _initial_mismatch_vector(channel.n)
    initial_bound = _bound_from_mismatches(initial_mismatches, channel)
    heap: list[tuple[float, int, int, _PrefixNode]] = []
    serial = 0
    root = _PrefixNode(initial_bound, 0, 0, initial_mismatches)
    heapq.heappush(heap, (-root.bound, 0, serial, root))
    work = WorkVector(generator_pushes=1, peak_frontier=1)
    best_index: int | None = None
    best_score = -1.0
    certified = False
    frontier_bound = initial_bound

    while heap and work.prefix_nodes < max_nodes:
        neg_bound, _, _, node = heapq.heappop(heap)
        work.generator_pops += 1
        frontier_bound = -float(neg_bound)
        if best_index is not None and best_score > frontier_bound + tolerance:
            certified = True
            break
        work.prefix_nodes += 1
        if node.length == channel.n:
            work.terminal_candidates += 1
            work.membership_queries += 1
            message = code.message_index(node.prefix)
            if message is not None:
                score = deletion_likelihood_dp(node.prefix, received, channel)
                work.exact_score_calls += 1
                work.likelihood_symbol_ops += channel.n * 4
                if score > best_score + tolerance:
                    best_score = float(score)
                    best_index = int(message)
            continue

        position = node.length
        for bit in (0, 1):
            prefix = node.prefix | (bit << position)
            if use_prefix_feasibility:
                feasible, operations = code.prefix_feasible(prefix, position + 1)
                work.prefix_feasibility_ops += operations
                if not feasible:
                    continue
            mismatch_vector, terms = _extend_mismatch_vector(
                node.mismatch_vector,
                position,
                bit,
                y,
            )
            work.prefix_bound_terms += terms
            bound = _bound_from_mismatches(mismatch_vector, channel)
            work.bound_checks += 1
            if best_index is None or bound >= best_score - tolerance:
                serial += 1
                child = _PrefixNode(bound, position + 1, prefix, mismatch_vector)
                heapq.heappush(heap, (-bound, -(position + 1), serial, child))
                work.generator_pushes += 1
        work.peak_frontier = max(work.peak_frontier, len(heap))

    if not certified and best_index is not None:
        frontier_bound = -heap[0][0] if heap else 0.0
        if best_score > frontier_bound + tolerance:
            certified = True

    fallback = not certified
    decision = best_index
    if fallback:
        decision = int(reference_decision)
        work.fallback_count = 1
        work.add(reference_work)
    work.wall_seconds += time.perf_counter() - start
    work.peak_rss_bytes = max(work.peak_rss_bytes, int(psutil.Process().memory_info().rss))
    exact = decision is not None and int(decision) in ml_tie_set
    return DecodeResult(
        decision=decision,
        ml_tie_set=ml_tie_set,
        exact=exact,
        certified=certified,
        fallback_used=fallback,
        incumbent_score=max(0.0, best_score),
        residual_bound=max(0.0, frontier_bound),
        work=work,
        notes={
            "algorithm": "PREFIX_FIBER_L2" if use_prefix_feasibility else "PREFIX_FIBER_L0",
            "max_nodes": max_nodes,
        },
    )
