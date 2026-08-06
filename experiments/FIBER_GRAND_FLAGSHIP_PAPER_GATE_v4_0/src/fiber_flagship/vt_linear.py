from __future__ import annotations

import time
from dataclasses import dataclass

import psutil

from .codes import VTOracle
from .utils import binary_tuple, insert_bit
from .work import Work


@dataclass(frozen=True)
class VTLinearOutcome:
    word: int
    valid: bool
    work: Work


def vt_decode_single_deletion_linear(received: int, code: VTOracle) -> VTLinearOutcome:
    """Classical O(n) VT single-deletion reconstruction in little-endian order."""
    start = time.perf_counter()
    n = code.n
    y = binary_tuple(received, n - 1)
    weighted = sum((i + 1) * y[i] for i in range(n - 1))
    weight = sum(y)
    syndrome = (code.syndrome_value - weighted) % (n + 1)
    work = Work()

    if syndrome <= weight:
        suffix_ones = weight
        position = 0
        for position in range(n):
            work.vt_candidate_checks += 1
            if suffix_ones == syndrome:
                break
            if position < n - 1:
                suffix_ones -= y[position]
        word = insert_bit(received, position, 0, n - 1)
    else:
        zeros_left_target = syndrome - weight - 1
        zeros_left = 0
        position = 0
        for position in range(n):
            work.vt_candidate_checks += 1
            if zeros_left == zeros_left_target:
                break
            if position < n - 1 and y[position] == 0:
                zeros_left += 1
        word = insert_bit(received, position, 1, n - 1)

    valid, ops = code.is_codeword(word)
    work.membership_queries = 1
    work.syndrome_bitops = ops
    work.wall_seconds = time.perf_counter() - start
    work.peak_rss_bytes = int(psutil.Process().memory_info().rss)
    return VTLinearOutcome(word=word, valid=valid, work=work)
