from __future__ import annotations

import heapq
import itertools
import math
import time
from dataclasses import dataclass
from typing import Callable, Iterable, Iterator, Sequence

import numpy as np
import psutil

from .channels import DeletionChannel, InsertionChannel
from .codes import BinaryCode
from .likelihood import deletion_likelihood_dp, one_insertion_likelihood
from .models import DecodeResult, WorkVector
from .utils import binary_tuple


def masks_of_weight(length: int, weight: int) -> Iterator[int]:
    if weight < 0 or weight > length:
        return
    for positions in itertools.combinations(range(length), weight):
        mask = 0
        for position in positions:
            mask |= 1 << position
        yield mask


def insert_hidden_bits(base: int, base_length: int, positions: Sequence[int], hidden: int) -> int:
    position_set = set(int(v) for v in positions)
    result = 0
    base_index = 0
    hidden_index = 0
    total_length = base_length + len(positions)
    for output_position in range(total_length):
        if output_position in position_set:
            bit = (int(hidden) >> hidden_index) & 1
            hidden_index += 1
        else:
            bit = (int(base) >> base_index) & 1
            base_index += 1
        result |= bit << output_position
    return result


def remove_bit(word: int, position: int, length: int) -> int:
    lower_mask = (1 << position) - 1
    lower = int(word) & lower_mask
    upper = int(word) >> (position + 1)
    return lower | (upper << position)


@dataclass
class _StreamItem:
    probability: float
    error_mask: int
    hidden_mask: int


class _ShellStream:
    def __init__(
        self,
        error_length: int,
        hidden_bits: int,
        probability_factor: float,
        p: float,
    ) -> None:
        self.error_length = int(error_length)
        self.hidden_bits = int(hidden_bits)
        self.factor = float(probability_factor)
        self.p = float(p)
        self._iterator = self._items()

    def _items(self) -> Iterator[_StreamItem]:
        weights = [0] if self.p == 0.0 else range(self.error_length + 1)
        for weight in weights:
            probability = self.factor * (
                1.0 if self.p == 0.0 else (self.p ** weight) * ((1.0 - self.p) ** (self.error_length - weight))
            )
            if probability <= 0.0:
                continue
            for error_mask in masks_of_weight(self.error_length, weight):
                for hidden_mask in range(1 << self.hidden_bits):
                    yield _StreamItem(float(probability), int(error_mask), int(hidden_mask))

    def next(self) -> _StreamItem | None:
        return next(self._iterator, None)


@dataclass(frozen=True)
class HistoryModel:
    name: str
    n: int
    stream_count: int
    streams: tuple[_ShellStream, ...]
    candidate_from_item: Callable[[int, _StreamItem], int]
    exact_score: Callable[[int], float]
    candidate_build_cost: int


def deletion_history_model(received: int, channel: DeletionChannel) -> HistoryModel:
    n = channel.n
    t = channel.deletions
    m = n - t
    received_mask = int(received)
    if t == 1 and channel.deletion_weights is not None:
        positions = [(index,) for index in range(n)]
        weights = list(float(v) for v in channel.deletion_weights)
    else:
        positions = list(itertools.combinations(range(n), t))
        weights = [1.0 / len(positions)] * len(positions)
    streams = tuple(
        _ShellStream(m, t, weights[index], channel.substitution_probability)
        for index in range(len(positions))
    )

    def candidate(stream_index: int, item: _StreamItem) -> int:
        base = received_mask ^ item.error_mask
        return insert_hidden_bits(base, m, positions[stream_index], item.hidden_mask)

    return HistoryModel(
        name=f"HISTORY_{channel.label}",
        n=n,
        stream_count=len(streams),
        streams=streams,
        candidate_from_item=candidate,
        exact_score=lambda word: deletion_likelihood_dp(word, received_mask, channel),
        candidate_build_cost=n,
    )


def insertion_history_model(received: int, channel: InsertionChannel) -> HistoryModel:
    n = channel.n
    y = binary_tuple(received, n + 1)
    q = (
        np.full(n + 1, 1.0 / (n + 1), dtype=float)
        if channel.insertion_position_weights is None
        else np.asarray(channel.insertion_position_weights, dtype=float)
    )
    factors = [
        float(q[j]) * (channel.inserted_one_probability if y[j] else 1.0 - channel.inserted_one_probability)
        for j in range(n + 1)
    ]
    streams = tuple(
        _ShellStream(n, 0, factors[j], channel.substitution_probability)
        for j in range(n + 1)
    )
    bases = [remove_bit(received, j, n + 1) for j in range(n + 1)]

    def candidate(stream_index: int, item: _StreamItem) -> int:
        return int(bases[stream_index]) ^ int(item.error_mask)

    return HistoryModel(
        name=f"HISTORY_{channel.label}",
        n=n,
        stream_count=len(streams),
        streams=streams,
        candidate_from_item=candidate,
        exact_score=lambda word: one_insertion_likelihood(word, received, channel),
        candidate_build_cost=max(1, n // 8 + 2),
    )


def _initialize_heap(model: HistoryModel) -> tuple[list[tuple[float, int, int, _StreamItem]], np.ndarray, int]:
    heap: list[tuple[float, int, int, _StreamItem]] = []
    eta = np.zeros(model.stream_count, dtype=float)
    pushes = 0
    for stream_index, stream in enumerate(model.streams):
        item = stream.next()
        if item is None:
            continue
        eta[stream_index] = item.probability
        heapq.heappush(heap, (-item.probability, stream_index, 0, item))
        pushes += 1
    return heap, eta, pushes


def history_fiber_decode(
    model: HistoryModel,
    code: BinaryCode,
    ml_tie_set: tuple[int, ...],
    reference_decision: int,
    reference_work: WorkVector,
    max_histories: int,
    tolerance: float = 1e-13,
    score_all_discovered: bool = False,
    record_trace: bool = False,
) -> DecodeResult:
    """Certified history-driven FIBER-GRAND with membership-first scoring.

    Noncodeword aggregate scores are not needed for correctness: the global
    unseen-candidate bound covers every undiscovered codeword, while discovered
    noncodewords are irrelevant competitors. Setting score_all_discovered=True
    reproduces the less efficient literal form of Algorithm 1 in the proposal.
    """
    start = time.perf_counter()
    heap, eta, pushes = _initialize_heap(model)
    residual = float(np.sum(eta))
    work = WorkVector(generator_pushes=pushes, peak_frontier=len(heap))
    seen: set[int] = set()
    scored_codewords: set[int] = set()
    best_index: int | None = None
    best_score = -1.0
    certified = False
    serial = 0
    trace: list[dict[str, float | int]] = []

    while heap:
        if work.history_components >= max_histories:
            break
        neg_probability, stream_index, _, item = heapq.heappop(heap)
        work.generator_pops += 1
        old_eta = float(eta[stream_index])
        next_item = model.streams[stream_index].next()
        if next_item is None:
            eta[stream_index] = 0.0
        else:
            eta[stream_index] = next_item.probability
            serial += 1
            heapq.heappush(
                heap,
                (-next_item.probability, stream_index, serial, next_item),
            )
            work.generator_pushes += 1
        residual += float(eta[stream_index]) - old_eta
        residual = max(0.0, residual)

        work.history_components += 1
        work.candidate_build_ops += model.candidate_build_cost
        candidate = model.candidate_from_item(stream_index, item)
        if candidate in seen:
            work.duplicate_histories += 1
        else:
            seen.add(candidate)
            work.distinct_candidates += 1
            work.peak_seen_candidates = max(work.peak_seen_candidates, len(seen))
            work.membership_queries += 1
            message = code.message_index(candidate)
            if score_all_discovered:
                score = model.exact_score(candidate)
                work.exact_score_calls += 1
                work.likelihood_symbol_ops += model.n * 4
            elif message is not None and candidate not in scored_codewords:
                score = model.exact_score(candidate)
                scored_codewords.add(candidate)
                work.exact_score_calls += 1
                work.likelihood_symbol_ops += model.n * 4
            else:
                score = None
            if message is not None and score is not None and score > best_score + tolerance:
                best_score = float(score)
                best_index = int(message)

        work.bound_checks += 1
        work.peak_frontier = max(work.peak_frontier, len(heap))
        if record_trace and (
            len(trace) < 50
            or work.history_components in (10, 25, 50, 100, 250, 500, 1000, 2500, 5000)
        ):
            trace.append(
                {
                    "histories": work.history_components,
                    "distinct_candidates": work.distinct_candidates,
                    "best_score": max(0.0, best_score),
                    "unseen_bound": residual,
                }
            )
        if best_index is not None and best_score > residual + tolerance:
            certified = True
            break

    fallback = not certified
    decision = best_index
    if fallback:
        decision = int(reference_decision)
        work.fallback_count = 1
        work.add(reference_work)
    elapsed = time.perf_counter() - start
    work.wall_seconds += elapsed
    work.peak_rss_bytes = max(work.peak_rss_bytes, int(psutil.Process().memory_info().rss))
    exact = decision is not None and int(decision) in ml_tie_set
    return DecodeResult(
        decision=decision,
        ml_tie_set=ml_tie_set,
        exact=exact,
        certified=certified,
        fallback_used=fallback,
        incumbent_score=max(0.0, best_score),
        residual_bound=residual,
        work=work,
        notes={
            "model": model.name,
            "score_all_discovered": score_all_discovered,
            "stream_count": model.stream_count,
            "max_histories": max_histories,
            "trace": trace if record_trace else None,
        },
    )


def pathwise_first_hit(
    model: HistoryModel,
    code: BinaryCode,
    max_histories: int,
) -> tuple[int | None, WorkVector]:
    heap, _, pushes = _initialize_heap(model)
    work = WorkVector(generator_pushes=pushes, peak_frontier=len(heap))
    serial = 0
    while heap and work.history_components < max_histories:
        _, stream_index, _, item = heapq.heappop(heap)
        work.generator_pops += 1
        next_item = model.streams[stream_index].next()
        if next_item is not None:
            serial += 1
            heapq.heappush(heap, (-next_item.probability, stream_index, serial, next_item))
            work.generator_pushes += 1
        work.history_components += 1
        work.candidate_build_ops += model.candidate_build_cost
        candidate = model.candidate_from_item(stream_index, item)
        work.membership_queries += 1
        message = code.message_index(candidate)
        if message is not None:
            return int(message), work
    return None, work
