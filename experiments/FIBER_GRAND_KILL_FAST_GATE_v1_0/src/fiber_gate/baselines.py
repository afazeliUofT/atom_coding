from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import psutil

from .channels import DeletionChannel, InsertionChannel
from .codes import BinaryCode
from .likelihood import (
    deletion_likelihood_vectorized,
    exact_ml_indices,
    insertion_likelihood_vectorized,
)
from .models import WorkVector


@dataclass
class ExactReference:
    decision: int
    tie_set: tuple[int, ...]
    scores: np.ndarray
    work: WorkVector
    name: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision": self.decision,
            "tie_set": list(self.tie_set),
            "name": self.name,
            "work": self.work.to_dict(),
        }


def exhaustive_deletion_ml(
    code: BinaryCode,
    received: int,
    channel: DeletionChannel,
) -> ExactReference:
    start = time.perf_counter()
    scores, operations = deletion_likelihood_vectorized(code.codewords_array, received, channel)
    ties = exact_ml_indices(scores)
    elapsed = time.perf_counter() - start
    work = WorkVector(
        direct_candidate_scores=code.size,
        direct_likelihood_ops=operations,
        wall_seconds=elapsed,
        peak_rss_bytes=int(psutil.Process().memory_info().rss),
    )
    return ExactReference(ties[0], ties, scores, work, "EXHAUSTIVE_CODEWORD_ML")


def exhaustive_insertion_ml(
    code: BinaryCode,
    received: int,
    channel: InsertionChannel,
) -> ExactReference:
    start = time.perf_counter()
    scores, operations = insertion_likelihood_vectorized(code.codewords_array, received, channel)
    ties = exact_ml_indices(scores)
    elapsed = time.perf_counter() - start
    work = WorkVector(
        direct_candidate_scores=code.size,
        direct_likelihood_ops=operations,
        wall_seconds=elapsed,
        peak_rss_bytes=int(psutil.Process().memory_info().rss),
    )
    return ExactReference(ties[0], ties, scores, work, "EXHAUSTIVE_CODEWORD_ML")


def best_path_codeword_branch_bound(
    exact_scores: np.ndarray,
    path_scores: np.ndarray,
    stream_count: int,
) -> tuple[int, WorkVector]:
    """Exact codeword-side bound using aggregate <= stream_count * best path.

    This baseline intentionally receives all best-path scores. Computing them
    is charged for every codeword, so it represents a strong but codebook-side
    branch-and-bound rather than a free oracle.
    """
    start = time.perf_counter()
    order = np.argsort(-np.asarray(path_scores, dtype=float), kind="stable")
    work = WorkVector(
        direct_candidate_scores=len(order),
        direct_likelihood_ops=len(order),
        sort_comparisons_proxy=int(max(1, len(order)) * math.log2(max(2, len(order)))),
    )
    best_index = int(order[0])
    best_score = float(exact_scores[best_index])
    scanned = 0
    for position, index in enumerate(order):
        scanned += 1
        score = float(exact_scores[index])
        if score > best_score:
            best_score = score
            best_index = int(index)
        next_upper = 0.0
        if position + 1 < len(order):
            next_upper = stream_count * float(path_scores[order[position + 1]])
        if best_score > next_upper + 1e-14:
            break
    work.exact_score_calls = scanned
    work.wall_seconds = time.perf_counter() - start
    work.peak_rss_bytes = int(psutil.Process().memory_info().rss)
    return best_index, work
