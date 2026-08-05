from __future__ import annotations

import heapq
import math
import time
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import psutil

from .codes import BinaryLinearCode, QaryRandomCodebook, digits_to_int, int_to_digits
from .enumeration import ProductAtomEnumerator
from .metrics import is_bi_unambiguous
from .models import DecodeResult, Representation, WorkVector


class ScoreTracker:
    """Exact lazy max tracker with implicit zero scores for untouched candidates."""

    def __init__(self, size: int) -> None:
        self.size = int(size)
        self.scores = np.zeros(size, dtype=float)
        self.versions = np.zeros(size, dtype=np.int64)
        self.active = np.zeros(size, dtype=bool)
        self.active_count = 0
        self.heap: list[tuple[float, int, int]] = []
        self.pushes = 0
        self.pops = 0

    def update(self, indices: np.ndarray, delta: float) -> None:
        for raw_index in indices:
            index = int(raw_index)
            if not self.active[index]:
                self.active[index] = True
                self.active_count += 1
            self.scores[index] += delta
            self.versions[index] += 1
            heapq.heappush(
                self.heap,
                (-float(self.scores[index]), index, int(self.versions[index])),
            )
            self.pushes += 1

    def _pop_stale(self) -> None:
        while self.heap:
            neg_score, index, version = self.heap[0]
            if version == int(self.versions[index]) and math.isclose(
                -neg_score, float(self.scores[index]), rel_tol=0.0, abs_tol=1e-15
            ):
                return
            heapq.heappop(self.heap)
            self.pops += 1

    def top_two(self) -> tuple[int, float, float]:
        self._pop_stale()
        if not self.heap:
            return 0, 0.0, 0.0 if self.size > 1 else float("-inf")
        top = heapq.heappop(self.heap)
        self.pops += 1
        top_score = -top[0]
        top_index = top[1]
        self._pop_stale()
        second_updated = -self.heap[0][0] if self.heap else float("-inf")
        zero_candidate_exists = self.active_count < self.size
        second_score = max(second_updated, 0.0) if zero_candidate_exists else second_updated
        heapq.heappush(self.heap, top)
        self.pushes += 1
        return top_index, float(top_score), float(second_score)


def direct_ml_from_codeword_array(
    channel: np.ndarray,
    codewords: np.ndarray,
    received: Sequence[int],
    tie_tolerance: float = 1e-12,
) -> tuple[int, tuple[int, ...], np.ndarray, WorkVector]:
    w = np.asarray(channel, dtype=float)
    y = np.asarray(received, dtype=int)
    if codewords.shape[1] != len(y):
        raise ValueError("Codeword length and received length differ")
    start = time.perf_counter()
    log_w = np.log(np.maximum(w, 1e-300))
    ll = np.zeros(codewords.shape[0], dtype=float)
    for coordinate in range(len(y)):
        ll += log_w[codewords[:, coordinate], y[coordinate]]
    maximum = float(np.max(ll))
    ties = tuple(int(v) for v in np.flatnonzero(ll >= maximum - tie_tolerance))
    work = WorkVector(
        direct_likelihood_ops=int(codewords.shape[0] * len(y) + max(0, codewords.shape[0] - 1)),
        wall_seconds=time.perf_counter() - start,
        peak_rss_bytes=int(psutil.Process().memory_info().rss),
    )
    return ties[0], ties, ll, work


def syndrome_trellis_ml(
    channel: np.ndarray,
    code: BinaryLinearCode,
    received: Sequence[int],
    tie_tolerance: float = 1e-12,
) -> tuple[int, float, np.ndarray, WorkVector]:
    """Exact ML decoding by Viterbi dynamic programming over parity syndromes.

    The state after coordinate i is the partial syndrome contributed by x_0,...,x_i.
    The terminal zero-syndrome path is the best codeword. This is a strong exact
    reference for high-rate binary linear codes: O(n 2^(n-k)) state transitions.
    """
    w = np.asarray(channel, dtype=float)
    y = np.asarray(received, dtype=int)
    if len(y) != code.n:
        raise ValueError("Codeword length and received length differ")
    if w.shape[0] != 2:
        raise ValueError("Syndrome trellis currently supports binary-input channels")

    start = time.perf_counter()
    redundancy = code.n - code.k
    states = 1 << redundancy
    neg_inf = float("-inf")
    current = np.full(states, neg_inf, dtype=float)
    current[0] = 0.0
    parent_state = np.full((code.n, states), -1, dtype=np.int32)
    parent_bit = np.full((code.n, states), -1, dtype=np.int8)
    log_w = np.log(np.maximum(w, 1e-300))
    updates = 0

    for coordinate in range(code.n):
        nxt = np.full(states, neg_inf, dtype=float)
        syndrome_delta = int(code.syndrome_column_masks[coordinate])
        metric0 = float(log_w[0, y[coordinate]])
        metric1 = float(log_w[1, y[coordinate]])
        active_states = np.flatnonzero(np.isfinite(current))
        for raw_state in active_states:
            state = int(raw_state)
            base = float(current[state])
            candidate0 = base + metric0
            updates += 1
            if candidate0 > nxt[state] + tie_tolerance:
                nxt[state] = candidate0
                parent_state[coordinate, state] = state
                parent_bit[coordinate, state] = 0
            candidate_state = state ^ syndrome_delta
            candidate1 = base + metric1
            updates += 1
            if candidate1 > nxt[candidate_state] + tie_tolerance:
                nxt[candidate_state] = candidate1
                parent_state[coordinate, candidate_state] = state
                parent_bit[coordinate, candidate_state] = 1
        current = nxt

    if not np.isfinite(current[0]):
        raise AssertionError("Zero-syndrome terminal state is unreachable")
    state = 0
    word_bits = np.zeros(code.n, dtype=np.uint8)
    traceback_ops = 0
    for coordinate in range(code.n - 1, -1, -1):
        bit = int(parent_bit[coordinate, state])
        previous = int(parent_state[coordinate, state])
        if bit < 0 or previous < 0:
            raise AssertionError("Trellis traceback encountered an unset parent")
        word_bits[coordinate] = bit
        state = previous
        traceback_ops += 1
    if state != 0:
        raise AssertionError("Trellis traceback did not return to the zero initial syndrome")

    word = digits_to_int((int(v) for v in word_bits), 2)
    message = code.message_index(word)
    if message is None:
        raise AssertionError("Trellis produced a word outside the code")
    work = WorkVector(
        trellis_state_updates=updates,
        trellis_traceback_ops=traceback_ops,
        wall_seconds=time.perf_counter() - start,
        peak_rss_bytes=int(psutil.Process().memory_info().rss),
    )
    return int(message), float(current[0]), word_bits, work


def strongest_binary_exact_reference(
    channel: np.ndarray,
    code: BinaryLinearCode,
    received: Sequence[int],
    tie_tolerance: float = 1e-12,
) -> dict[str, object]:
    """Compute independent direct-ML oracle and the cheaper exact trellis reference."""
    direct_decision, direct_ties, direct_scores, direct_work = direct_ml_from_codeword_array(
        channel, code.codewords_array, received, tie_tolerance=tie_tolerance
    )
    trellis_decision, trellis_score, trellis_word, trellis_work = syndrome_trellis_ml(
        channel, code, received, tie_tolerance=tie_tolerance
    )
    if trellis_decision not in direct_ties:
        raise AssertionError(
            f"Syndrome-trellis decision {trellis_decision} disagrees with direct ML ties {direct_ties}"
        )
    direct_best = float(np.max(direct_scores))
    if not math.isclose(trellis_score, direct_best, rel_tol=0.0, abs_tol=max(tie_tolerance, 1e-10)):
        raise AssertionError(
            f"Syndrome-trellis score {trellis_score} disagrees with direct ML score {direct_best}"
        )
    direct_scalar = direct_work.scalar("balanced")
    trellis_scalar = trellis_work.scalar("balanced")
    if trellis_scalar < direct_scalar:
        name = "syndrome_trellis_ml"
        selected_work = trellis_work
    else:
        name = "direct_codeword_ml"
        selected_work = direct_work
    return {
        "decision": direct_decision,
        "tie_set": direct_ties,
        "scores": direct_scores,
        "direct_work": direct_work,
        "trellis_work": trellis_work,
        "selected_name": name,
        "selected_work": selected_work,
        "trellis_word": trellis_word,
    }


def one_shot_residual_decode(
    channel: np.ndarray,
    representation: Representation,
    received: int,
    order: Sequence[int] | None = None,
    tie_tolerance: float = 1e-12,
) -> DecodeResult:
    representation.verify(channel)
    m = channel.shape[0]
    schedule = list(range(representation.support_size)) if order is None else list(order)
    if sorted(schedule) != list(range(representation.support_size)):
        raise ValueError("order must be a permutation of atom indices")
    tracker = ScoreTracker(m)
    remaining = 1.0
    work = WorkVector(representation_atoms=representation.support_size)
    start = time.perf_counter()
    certified = False
    decision = 0
    for atom_index in schedule:
        atom = representation.maps[atom_index]
        weight = float(representation.weights[atom_index])
        work.atoms_processed += 1
        work.fiber_queries += 1
        remaining = max(0.0, remaining - weight)
        fiber = np.array([x for x in range(m) if atom(x) == received], dtype=np.int64)
        work.fiber_entries += len(fiber)
        work.score_updates += len(fiber)
        tracker.update(fiber, weight)
        decision, best, second = tracker.top_two()
        if best > second + remaining + tie_tolerance:
            certified = True
            break
    work.heap_pushes = tracker.pushes
    work.heap_pops = tracker.pops
    work.wall_seconds = time.perf_counter() - start
    work.peak_rss_bytes = int(psutil.Process().memory_info().rss)
    ml_values = channel[:, received]
    ml_max = float(np.max(ml_values))
    ml_ties = tuple(int(v) for v in np.flatnonzero(ml_values >= ml_max - tie_tolerance))
    exact = decision in ml_ties
    return DecodeResult(
        decision=decision,
        ml_tie_set=ml_ties,
        certified=certified,
        exact=exact,
        fallback_used=False,
        atoms_processed=work.atoms_processed,
        residual_mass=remaining,
        work=work,
        scores=tracker.scores.copy(),
    )


def binary_product_residual_decode(
    channel: np.ndarray,
    representation: Representation,
    code: BinaryLinearCode,
    received: Sequence[int],
    max_atoms: int,
    tie_tolerance: float = 1e-12,
    reference: dict[str, object] | None = None,
) -> DecodeResult:
    representation.verify(channel)
    if representation.input_size != 2:
        raise ValueError("Binary product decoder requires binary input atoms")
    y = tuple(int(v) for v in received)
    if len(y) != code.n:
        raise ValueError("Received blocklength does not match code")

    rep = representation.reduced().sorted_by_weight()
    enumerator = ProductAtomEnumerator(rep.weights, code.n)
    tracker = ScoreTracker(code.size)
    work = WorkVector(representation_atoms=rep.support_size)
    remaining = 1.0
    cumulative_mass = 0.0
    certified = False
    decision = 0
    start = time.perf_counter()

    for item in enumerator:
        if work.atoms_processed >= max_atoms:
            break
        work.atoms_processed += 1
        work.fiber_queries += 1
        work.inverse_symbol_ops += code.n
        cumulative_mass += item.probability
        remaining = max(0.0, 1.0 - cumulative_mass)

        fixed_mask = 0
        fixed_value = 0
        valid = True
        for coordinate, atom_index in enumerate(item.atom_indices):
            atom = rep.maps[atom_index]
            out0, out1 = atom.outputs
            target = y[coordinate]
            match0 = out0 == target
            match1 = out1 == target
            if not match0 and not match1:
                valid = False
                break
            if match0 ^ match1:
                fixed_mask |= 1 << coordinate
                if match1:
                    fixed_value |= 1 << coordinate
        if valid:
            cache_key = (fixed_mask, fixed_value & fixed_mask)
            cache_hit = cache_key in code._fiber_cache
            solution = code.fiber(fixed_mask, fixed_value)
            if not cache_hit:
                work.fiber_solver_bitops += solution.estimated_bitops
            fiber = solution.message_indices
            work.fiber_entries += len(fiber)
            work.score_updates += len(fiber)
            tracker.update(fiber, item.probability)

        decision, best, second = tracker.top_two()
        if best > second + remaining + tie_tolerance:
            certified = True
            break

    work.generator_pushes = enumerator.pushes
    work.generator_pops = enumerator.pops
    work.heap_pushes = tracker.pushes
    work.heap_pops = tracker.pops
    atom_elapsed = time.perf_counter() - start
    atom_peak = int(psutil.Process().memory_info().rss)

    if reference is None:
        reference = strongest_binary_exact_reference(
            channel, code, y, tie_tolerance=tie_tolerance
        )
    direct_decision = int(reference["decision"])
    direct_ties = tuple(int(v) for v in reference["tie_set"])
    selected_work = reference["selected_work"]
    fallback = not certified
    if fallback:
        decision = direct_decision
        work.fallback_count = 1
        work.direct_likelihood_ops += selected_work.direct_likelihood_ops
        work.trellis_state_updates += selected_work.trellis_state_updates
        work.trellis_traceback_ops += selected_work.trellis_traceback_ops
        work.wall_seconds = atom_elapsed + selected_work.wall_seconds
        work.peak_rss_bytes = max(atom_peak, selected_work.peak_rss_bytes)
    else:
        work.wall_seconds = atom_elapsed
        work.peak_rss_bytes = atom_peak
    exact = decision in direct_ties
    return DecodeResult(
        decision=decision,
        ml_tie_set=direct_ties,
        certified=certified,
        exact=exact,
        fallback_used=fallback,
        atoms_processed=work.atoms_processed,
        residual_mass=remaining,
        work=work,
        scores=tracker.scores.copy(),
        notes={
            "direct_decision": direct_decision,
            "direct_work": reference["direct_work"].to_dict(),
            "trellis_work": reference["trellis_work"].to_dict(),
            "reference_name": reference["selected_name"],
            "reference_work": selected_work.to_dict(),
            "max_atoms": max_atoms,
            "generator_exhausted": len(enumerator.heap) == 0,
        },
    )


def qary_reversible_first_hit_decode(
    channel: np.ndarray,
    representation: Representation,
    code: QaryRandomCodebook,
    received: Sequence[int],
    max_atoms: int,
    tie_tolerance: float = 1e-12,
) -> DecodeResult:
    representation.verify(channel)
    if not is_bi_unambiguous(representation):
        raise ValueError("First-hit decoder requires a bi-unambiguous representation")
    y = tuple(int(v) for v in received)
    if len(y) != code.n:
        raise ValueError("Received blocklength does not match code")

    rep = representation.reduced().sorted_by_weight()
    inverse_maps: list[dict[int, int]] = []
    for atom in rep.maps:
        inverse_maps.append({out: x for x, out in enumerate(atom.outputs)})
    enumerator = ProductAtomEnumerator(rep.weights, code.n)
    work = WorkVector(representation_atoms=rep.support_size)
    decision: int | None = None
    cumulative_mass = 0.0
    start = time.perf_counter()

    for item in enumerator:
        if work.atoms_processed >= max_atoms:
            break
        work.atoms_processed += 1
        work.membership_queries += 1
        work.inverse_symbol_ops += code.n
        cumulative_mass += item.probability
        x_digits = []
        valid = True
        for coordinate, atom_index in enumerate(item.atom_indices):
            inverse = inverse_maps[atom_index]
            target = y[coordinate]
            if target not in inverse:
                valid = False
                break
            x_digits.append(inverse[target])
        if not valid:
            continue
        word = digits_to_int(x_digits, code.q)
        member = code.message_index(word)
        if member is not None:
            decision = member
            break

    work.generator_pushes = enumerator.pushes
    work.generator_pops = enumerator.pops
    direct_decision, direct_ties, _, direct_work = direct_ml_from_codeword_array(
        channel, code.digits, y, tie_tolerance=tie_tolerance
    )
    fallback = decision is None
    if fallback:
        decision = direct_decision
        work.fallback_count = 1
        work.direct_likelihood_ops += direct_work.direct_likelihood_ops
    exact = int(decision) in direct_ties
    work.wall_seconds = time.perf_counter() - start
    work.peak_rss_bytes = int(psutil.Process().memory_info().rss)
    return DecodeResult(
        decision=int(decision),
        ml_tie_set=direct_ties,
        certified=not fallback,
        exact=exact,
        fallback_used=fallback,
        atoms_processed=work.atoms_processed,
        residual_mass=max(0.0, 1.0 - cumulative_mass),
        work=work,
        scores=None,
        notes={"direct_work": direct_work.to_dict(), "max_atoms": max_atoms},
    )
