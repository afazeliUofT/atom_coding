from __future__ import annotations

import itertools
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from .codelengths import kt_log2_probability, lz78_codelength_bits
from .markov_types import MarkovTypeAtlas, transition_counts
from .utils import binary_tuple, logsumexp2, write_json


def _upper_level_log2_ranks_from_lengths(lengths: np.ndarray) -> np.ndarray:
    order = np.argsort(lengths, kind="mergesort")
    result = np.empty(len(lengths), dtype=float)
    cursor = 0
    cumulative = 0
    while cursor < len(order):
        start = cursor
        value = lengths[order[cursor]]
        while cursor < len(order) and lengths[order[cursor]] == value:
            cursor += 1
        cumulative += cursor - start
        result[order[start:cursor]] = math.log2(cumulative)
    return result


def _upper_level_log2_ranks_from_scores(scores: np.ndarray, tolerance: float = 1e-12) -> np.ndarray:
    order = np.argsort(-scores, kind="mergesort")
    result = np.empty(len(scores), dtype=float)
    cursor = 0
    cumulative = 0
    while cursor < len(order):
        start = cursor
        reference = scores[order[cursor]]
        while cursor < len(order) and abs(scores[order[cursor]] - reference) <= tolerance:
            cursor += 1
        cumulative += cursor - start
        result[order[start:cursor]] = math.log2(cumulative)
    return result


def _order0_target_fitted_type_ranks(atlas: MarkovTypeAtlas) -> np.ndarray:
    output = np.empty(len(atlas.types), dtype=float)
    cache: dict[int, float] = {}
    for index, target in enumerate(atlas.types):
        ones = target.n1
        if ones in cache:
            output[index] = cache[ones]
            continue
        p = ones / atlas.n
        if p <= 0.0:
            scores = np.where(atlas.n1 == 0, 0.0, float("-inf"))
        elif p >= 1.0:
            scores = np.where(atlas.n0 == 0, 0.0, float("-inf"))
        else:
            scores = atlas.n1 * math.log2(p) + atlas.n0 * math.log2(1.0 - p)
        rank = atlas.rank_for_score_vector(scores, index)
        cache[ones] = rank
        output[index] = rank
    return output


def _order1_target_fitted_type_ranks(atlas: MarkovTypeAtlas) -> np.ndarray:
    output = np.empty(len(atlas.types), dtype=float)
    for index, target in enumerate(atlas.types):
        # Maximum-likelihood transition parameters for the target type.
        out0 = target.n00 + target.n01
        out1 = target.n10 + target.n11
        p01 = target.n01 / out0 if out0 else 0.5
        p10 = target.n10 / out1 if out1 else 0.5
        eps = 1e-15
        p01 = min(max(p01, eps), 1.0 - eps)
        p10 = min(max(p10, eps), 1.0 - eps)
        scores = atlas.markov_scores(p01, p10, pi1=0.5)
        output[index] = atlas.rank_for_score_vector(scores, index)
    return output


def deterministic_tie_counterexample(n: int) -> dict[str, Any]:
    ambient = 1 << n
    lengths = np.array([lz78_codelength_bits(binary_tuple(value, n)) for value in range(ambient)], dtype=int)
    ranks = _upper_level_log2_ranks_from_lengths(lengths)
    target = int(np.argmax(ranks))
    universal_rank = float(ranks[target])
    # A one-state uniform assignment gives every sequence equal probability. The
    # unspecified deterministic tie order can put the selected target first.
    comparator_rank = 0.0
    return {
        "n": n,
        "target_integer": target,
        "target_bits_lsb_first": list(binary_tuple(target, n)),
        "lz_codelength": int(lengths[target]),
        "log2_lz_upper_level_rank": universal_rank,
        "log2_uniform_comparator_rank_under_target_first_tie": comparator_rank,
        "gap_bits": universal_rank - comparator_rank,
        "gap_per_symbol": (universal_rank - comparator_rank) / n,
        "structural_failure": bool(universal_rank >= n - 1e-12),
        "interpretation": (
            "The probability assignment is a valid one-state finite-state model. "
            "Because all sequences tie, an unrestricted deterministic tie order can place this target first."
        ),
    }


def exact_rank_atlas(output_dir: Path, blocklengths: Sequence[int]) -> dict[str, Any]:
    summary_rows: list[dict[str, Any]] = []
    witness_rows: list[dict[str, Any]] = []
    for n in blocklengths:
        ambient = 1 << n
        sequences = [binary_tuple(value, n) for value in range(ambient)]
        lz_lengths = np.fromiter((lz78_codelength_bits(bits) for bits in sequences), dtype=np.int32, count=ambient)
        kt0_scores = np.fromiter((kt_log2_probability(bits, order=0) for bits in sequences), dtype=float, count=ambient)
        kt1_scores = np.fromiter((kt_log2_probability(bits, order=1) for bits in sequences), dtype=float, count=ambient)
        lz_ranks = _upper_level_log2_ranks_from_lengths(lz_lengths)
        kt0_ranks = _upper_level_log2_ranks_from_scores(kt0_scores)
        kt1_ranks = _upper_level_log2_ranks_from_scores(kt1_scores)

        atlas = MarkovTypeAtlas.build(n)
        fitted0_by_type = _order0_target_fitted_type_ranks(atlas)
        fitted1_by_type = _order1_target_fitted_type_ranks(atlas)
        fitted0 = np.empty(ambient, dtype=float)
        fitted1 = np.empty(ambient, dtype=float)
        for value, bits in enumerate(sequences):
            index = atlas.key_to_index[transition_counts(bits)]
            fitted0[value] = fitted0_by_type[index]
            fitted1[value] = fitted1_by_type[index]
        best_comparator = np.minimum(fitted0, fitted1)
        regret = lz_ranks - best_comparator
        worst = int(np.argmax(regret))
        median_regret = float(np.median(regret))
        positive_fraction = float(np.mean(regret > 0.0))
        summary_rows.append(
            {
                "n": n,
                "ambient": ambient,
                "lz_distinct_lengths": int(len(np.unique(lz_lengths))),
                "worst_log2_rank_regret_vs_target_fitted_order01": float(regret[worst]),
                "worst_normalized_rank_regret": float(regret[worst] / n),
                "median_log2_rank_regret": median_regret,
                "fraction_sequences_lz_rank_worse": positive_fraction,
                "mean_log2_lz_rank": float(np.mean(lz_ranks)),
                "mean_log2_kt1_rank": float(np.mean(kt1_ranks)),
                "mean_log2_target_fitted_rank": float(np.mean(best_comparator)),
            }
        )
        witness_rows.append(
            {
                "n": n,
                "target_integer": worst,
                "target_bits_lsb_first": "".join(str(v) for v in sequences[worst]),
                "lz_length": int(lz_lengths[worst]),
                "log2_lz_rank": float(lz_ranks[worst]),
                "log2_kt0_rank": float(kt0_ranks[worst]),
                "log2_kt1_rank": float(kt1_ranks[worst]),
                "log2_target_fitted_order0_rank": float(fitted0[worst]),
                "log2_target_fitted_order1_rank": float(fitted1[worst]),
                "regret_bits": float(regret[worst]),
            }
        )
    frame = pd.DataFrame(summary_rows)
    frame.to_csv(output_dir / "02_repaired_rank_atlas.csv", index=False)
    pd.DataFrame(witness_rows).to_csv(output_dir / "02_repaired_rank_witnesses.csv", index=False)
    return {
        "blocklengths": list(blocklengths),
        "rows": summary_rows,
        "largest_n_worst_normalized_regret": float(summary_rows[-1]["worst_normalized_rank_regret"]),
        "largest_n_median_regret_bits": float(summary_rows[-1]["median_log2_rank_regret"]),
        "interpretation": (
            "Tie-invariant upper-level ranks remove the immediate deterministic tie pathology. "
            "This atlas is a finite counterexample search, not a proof of an individual-sequence theorem."
        ),
    }


def random_coding_formula_audit() -> dict[str, Any]:
    cases: list[dict[str, Any]] = []
    max_power_gap = 0.0
    for n in range(2, 6):
        N = 1 << n
        for M in range(2, min(5, N)):
            for rank in range(1, N + 1):
                bad = rank - 1
                total_subsets = math.comb(N - 1, M - 1)
                good_pool = (N - 1) - bad
                no_error_subsets = math.comb(good_pool, M - 1) if good_pool >= M - 1 else 0
                exact_without_replacement = 1.0 - no_error_subsets / total_subsets
                with_replacement = 1.0 - (1.0 - bad / (N - 1)) ** (M - 1)
                proposal_power = 1.0 - (1.0 - bad / N) ** (M - 1)
                max_power_gap = max(max_power_gap, abs(exact_without_replacement - proposal_power))
                # Exhaustive subset enumeration for a representative subset of cases.
                if n <= 4 and rank in (1, max(1, N // 2), N):
                    errors = 0
                    total = 0
                    universe = list(range(1, N))
                    # Define the bad competitor set abstractly by cardinality; symmetry makes labels irrelevant.
                    bad_set = set(universe[:bad])
                    for subset in itertools.combinations(universe, M - 1):
                        total += 1
                        errors += int(any(value in bad_set for value in subset))
                    observed = errors / total
                    if not math.isclose(observed, exact_without_replacement, abs_tol=1e-15):
                        raise AssertionError("Hypergeometric audit mismatch")
                cases.append(
                    {
                        "n": n,
                        "N": N,
                        "M": M,
                        "rank": rank,
                        "exact_without_replacement": exact_without_replacement,
                        "with_replacement_excluding_transmitted": with_replacement,
                        "proposal_denominator_N_power": proposal_power,
                    }
                )
    return {
        "cases_checked": len(cases),
        "maximum_absolute_gap_proposal_power_vs_exact_without_replacement": max_power_gap,
        "exact_law": "1 - C(N-rank, M-1)/C(N-1, M-1)",
        "conclusion": (
            "The proposal's displayed equality is ensemble-dependent and is not the exact distinct-codeword law. "
            "Its exponent-level reduction remains plausible after the ensemble is specified correctly."
        ),
    }


def regret_cancellation_counterexample(amplitude_bits: float = 40.0) -> dict[str, Any]:
    differences = [amplitude_bits, -amplitude_bits]
    signed_regret = sum(differences)
    positive_regret = sum(max(0.0, value) for value in differences)
    rank_factors = [2.0**value for value in differences]
    # A bounded error probability can saturate on the bad frame and cannot be cancelled by an improvement below zero.
    baseline_errors = [2.0 ** (-amplitude_bits / 2.0), 0.5]
    candidate_errors = [min(1.0, rank_factors[0] * baseline_errors[0]), min(1.0, rank_factors[1] * baseline_errors[1])]
    return {
        "frame_codelength_differences_bits": differences,
        "signed_total_regret_bits": signed_regret,
        "positive_part_regret_bits": positive_regret,
        "rank_factors": rank_factors,
        "baseline_errors": baseline_errors,
        "candidate_errors": candidate_errors,
        "baseline_mean_error": sum(baseline_errors) / 2.0,
        "candidate_mean_error": sum(candidate_errors) / 2.0,
        "structural_failure": bool(signed_regret == 0.0 and sum(candidate_errors) > sum(baseline_errors)),
        "required_repair": (
            "Replace signed cumulative regret by one-sided positive-part regret, a uniform pointwise bound, "
            "or an exponential-moment control tailored to rank/error transfer."
        ),
    }


def run_individual_sequence_gate(output_dir: Path, blocklengths: Sequence[int]) -> dict[str, Any]:
    counterexample = deterministic_tie_counterexample(max(blocklengths))
    formula = random_coding_formula_audit()
    regret = regret_cancellation_counterexample()
    atlas = exact_rank_atlas(output_dir, blocklengths)
    payload = {
        "original_deterministic_conjecture": {
            "pass": False,
            "counterexample": counterexample,
            "reason": (
                "An unrestricted deterministic tie order for a one-state uniform finite-state assignment can rank "
                "a target first, whereas its LZ upper-level rank can be the full ambient size."
            ),
        },
        "per_sequence_random_coding_reduction": formula,
        "session_regret_transfer": {
            "pass_as_stated": False,
            "counterexample": regret,
        },
        "repaired_level_set_atlas": atlas,
        "surviving_theory_candidate": (
            "A tie-invariant or randomized guessing theorem may survive, but its novelty must be audited against "
            "existing finite-state individual-sequence randomized guessing results."
        ),
    }
    write_json(output_dir / "02_individual_sequence_gate.json", payload)
    return payload
