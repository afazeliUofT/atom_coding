from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from .codelengths import entropy_rate_markov, renyi_half_rate_markov
from .markov_types import MarkovTypeAtlas
from .processes import DEFAULT_REGIMES, MarkovRegime, fit_markov, sample_markov, transition_counts_array
from .utils import random_coding_bler_from_log2_level_rank, write_json


def _stationary_metrics_for_atlas(atlas: MarkovTypeAtlas, regime: MarkovRegime) -> dict[str, np.ndarray]:
    score_vectors = {
        "ORACLE_MARKOV": atlas.markov_scores(regime.p01, regime.p10),
        "KT1_UNIVERSAL": atlas.kt_order1_scores(),
        "EMPIRICAL_ML1": atlas.empirical_markov_scores(),
        "KT0_MEMORYLESS": atlas.kt_order0_scores(),
    }
    return {name: atlas.level_set_log2_ranks(scores) for name, scores in score_vectors.items()}


def _rank_under_fitted_training(atlas: MarkovTypeAtlas, target_index: int, training: Sequence[int]) -> float:
    p01, p10 = fit_markov(training, pseudocount=0.5)
    scores = atlas.markov_scores(p01, p10)
    return atlas.rank_for_score_vector(scores, target_index)


def run_stationary_rank_benchmark(
    output_dir: Path,
    rng: np.random.Generator,
    blocklengths: Sequence[int],
    rates: Sequence[float],
    trials: int,
    training_lengths: Sequence[int],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for n in blocklengths:
        atlas = MarkovTypeAtlas.build(n)
        for regime in DEFAULT_REGIMES:
            rank_vectors = _stationary_metrics_for_atlas(atlas, regime)
            sequences = [sample_markov(n, regime.p01, regime.p10, rng) for _ in range(trials)]
            trainings = {
                length: [sample_markov(length, regime.p01, regime.p10, rng) for _ in range(trials)]
                for length in training_lengths
            }
            for trial, sequence in enumerate(sequences):
                target_index = atlas.index_of_sequence(sequence)
                ranks: dict[str, float] = {name: float(vector[target_index]) for name, vector in rank_vectors.items()}
                for length in training_lengths:
                    ranks[f"FIT_{length}"] = _rank_under_fitted_training(atlas, target_index, trainings[length][trial])
                oracle = ranks["ORACLE_MARKOV"]
                for rate in rates:
                    for metric, log_rank in ranks.items():
                        rows.append(
                            {
                                "regime": regime.name,
                                "structured": regime.structured,
                                "n": n,
                                "rate": rate,
                                "trial": trial,
                                "metric": metric,
                                "log2_level_rank": log_rank,
                                "normalized_rank": log_rank / n,
                                "oracle_overhead_bits": log_rank - oracle,
                                "effective_log2_queries": min(log_rank, n * (1.0 - rate)),
                                "random_coding_bler_proxy": random_coding_bler_from_log2_level_rank(log_rank, n, rate),
                            }
                        )
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "04_stationary_rank_trials.csv.gz", index=False, compression="gzip")

    summary_rows: list[dict[str, Any]] = []
    for (regime, n, rate, metric), group in frame.groupby(["regime", "n", "rate", "metric"]):
        summary_rows.append(
            {
                "regime": regime,
                "n": int(n),
                "rate": float(rate),
                "metric": metric,
                "mean_log2_rank": float(group["log2_level_rank"].mean()),
                "median_log2_rank": float(group["log2_level_rank"].median()),
                "p95_log2_rank": float(group["log2_level_rank"].quantile(0.95)),
                "mean_normalized_rank": float(group["normalized_rank"].mean()),
                "median_oracle_overhead_bits": float(group["oracle_overhead_bits"].median()),
                "p95_oracle_overhead_bits": float(group["oracle_overhead_bits"].quantile(0.95)),
                "mean_effective_log2_queries": float(group["effective_log2_queries"].mean()),
                "mean_bler_proxy": float(group["random_coding_bler_proxy"].mean()),
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_dir / "04_stationary_rank_summary.csv", index=False)

    largest_n = max(blocklengths)
    largest_rate = max(rates)
    details: dict[str, Any] = {}
    structured_pass_count = 0
    for regime in DEFAULT_REGIMES:
        subset = summary[(summary["regime"] == regime.name) & (summary["n"] == largest_n) & (summary["rate"] == largest_rate)]
        by_metric = {row["metric"]: row for _, row in subset.iterrows()}
        oracle = by_metric["ORACLE_MARKOV"]
        kt1 = by_metric["KT1_UNIVERSAL"]
        kt0 = by_metric["KT0_MEMORYLESS"]
        fit_short = by_metric.get(f"FIT_{min(training_lengths)}")
        fit_long = by_metric.get(f"FIT_{max(training_lengths)}")
        universal_overhead_norm = float(kt1["median_oracle_overhead_bits"] / largest_n)
        memory_gain_norm = float((kt0["median_log2_rank"] - kt1["median_log2_rank"]) / largest_n)
        short_fit_advantage_norm = float((kt1["median_log2_rank"] - fit_short["median_log2_rank"]) / largest_n) if fit_short is not None else float("nan")
        long_fit_advantage_norm = float((kt1["median_log2_rank"] - fit_long["median_log2_rank"]) / largest_n) if fit_long is not None else float("nan")
        query_budget_log2 = math.log2(1.0e4)
        query_kill_log2 = math.log2(1.0e6)
        median_query_budget_pass = float(kt1["median_log2_rank"]) <= query_budget_log2
        persistent_query_kill = float(kt1["median_log2_rank"]) > query_kill_log2
        passed = bool(
            regime.structured
            and universal_overhead_norm <= 0.04
            and memory_gain_norm >= 0.03
            and short_fit_advantage_norm <= 0.03
            and median_query_budget_pass
        )
        structured_pass_count += int(passed)
        details[regime.name] = {
            "structured": regime.structured,
            "entropy_rate": entropy_rate_markov(regime.p01, regime.p10),
            "renyi_half_rate": renyi_half_rate_markov(regime.p01, regime.p10),
            "median_universal_oracle_overhead_per_symbol": universal_overhead_norm,
            "median_memoryless_minus_universal_rank_per_symbol": memory_gain_norm,
            "median_universal_minus_short_fit_rank_per_symbol": short_fit_advantage_norm,
            "median_universal_minus_long_fit_rank_per_symbol": long_fit_advantage_norm,
            "universal_median_log2_rank": float(kt1["median_log2_rank"]),
            "universal_median_query_count_approx": float(2.0 ** min(float(kt1["median_log2_rank"]), 1023.0)),
            "proposal_median_query_budget_1e4_pass": median_query_budget_pass,
            "proposal_query_kill_1e6_triggered": persistent_query_kill,
            "oracle_median_log2_rank": float(oracle["median_log2_rank"]),
            "pass_practical_stationary_signal": passed,
        }
    payload = {
        "largest_n": largest_n,
        "largest_rate": largest_rate,
        "trials_per_regime_n": trials,
        "training_lengths": list(training_lengths),
        "structured_passing_regimes": structured_pass_count,
        "regime_details": details,
        "pass": structured_pass_count >= 2,
        "rank_quality_pass_without_absolute_query_budget": sum(
            int(row["structured"] and row["median_universal_oracle_overhead_per_symbol"] <= 0.04
                and row["median_memoryless_minus_universal_rank_per_symbol"] >= 0.03)
            for row in details.values()
        ) >= 2,
        "novelty_boundary": (
            "A pass confirms finite-length usefulness of universal Markov ranking, not field-defining novelty; "
            "stochastic finite-state universal noise guessing and training comparisons already exist."
        ),
    }
    write_json(output_dir / "04_stationary_rank_gate.json", payload)
    return payload


def _beta_score_vector(atlas: MarkovTypeAtlas, prior: np.ndarray) -> np.ndarray:
    return atlas.kt_order1_scores(prior_counts=prior)


def _update_prior(prior: np.ndarray, sequence: Sequence[int], discount: float = 1.0) -> np.ndarray:
    n00, n01, n10, n11 = transition_counts_array(sequence)
    counts = np.array([n00, n01, n10, n11], dtype=float)
    return 0.5 + discount * (prior - 0.5) + counts


def run_nonstationary_adaptation_benchmark(
    output_dir: Path,
    rng: np.random.Generator,
    n: int,
    rate: float,
    frames: int,
    switch_every: int,
    sessions: int,
    discount: float,
) -> dict[str, Any]:
    atlas = MarkovTypeAtlas.build(n)
    regime_a = DEFAULT_REGIMES[0]
    regime_b = DEFAULT_REGIMES[2]
    rows: list[dict[str, Any]] = []
    for session in range(sessions):
        cumulative_prior = np.full(4, 0.5, dtype=float)
        discounted_prior = np.full(4, 0.5, dtype=float)
        fixed_training = sample_markov(max(64, n), regime_a.p01, regime_a.p10, rng)
        fixed_p01, fixed_p10 = fit_markov(fixed_training)
        fixed_scores = atlas.markov_scores(fixed_p01, fixed_p10)
        fixed_ranks = atlas.level_set_log2_ranks(fixed_scores)
        for frame_index in range(frames):
            regime = regime_a if (frame_index // switch_every) % 2 == 0 else regime_b
            sequence = sample_markov(n, regime.p01, regime.p10, rng)
            target = atlas.index_of_sequence(sequence)
            oracle_scores = atlas.markov_scores(regime.p01, regime.p10)
            oracle_rank = atlas.rank_for_score_vector(oracle_scores, target)
            cumulative_scores = _beta_score_vector(atlas, cumulative_prior)
            discounted_scores = _beta_score_vector(atlas, discounted_prior)
            metrics = {
                "ORACLE_SWITCHING": oracle_rank,
                "FIXED_INITIAL_FIT": float(fixed_ranks[target]),
                "CUMULATIVE_KT": atlas.rank_for_score_vector(cumulative_scores, target),
                "DISCOUNTED_KT": atlas.rank_for_score_vector(discounted_scores, target),
            }
            for metric, log_rank in metrics.items():
                rows.append(
                    {
                        "session": session,
                        "frame": frame_index,
                        "regime": regime.name,
                        "metric": metric,
                        "n": n,
                        "rate": rate,
                        "log2_level_rank": log_rank,
                        "oracle_overhead_bits": log_rank - oracle_rank,
                        "effective_log2_queries": min(log_rank, n * (1.0 - rate)),
                        "bler_proxy": random_coding_bler_from_log2_level_rank(log_rank, n, rate),
                    }
                )
            cumulative_prior = _update_prior(cumulative_prior, sequence, discount=1.0)
            discounted_prior = _update_prior(discounted_prior, sequence, discount=discount)
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "05_nonstationary_trials.csv.gz", index=False, compression="gzip")
    summary_rows: list[dict[str, Any]] = []
    for metric, group in frame.groupby("metric"):
        summary_rows.append(
            {
                "metric": metric,
                "mean_log2_rank": float(group["log2_level_rank"].mean()),
                "median_log2_rank": float(group["log2_level_rank"].median()),
                "p95_log2_rank": float(group["log2_level_rank"].quantile(0.95)),
                "mean_oracle_overhead_bits": float(group["oracle_overhead_bits"].mean()),
                "mean_effective_log2_queries": float(group["effective_log2_queries"].mean()),
                "mean_bler_proxy": float(group["bler_proxy"].mean()),
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_dir / "05_nonstationary_summary.csv", index=False)
    by_metric = {row["metric"]: row for _, row in summary.iterrows()}
    adaptive_gain = (
        float(by_metric["FIXED_INITIAL_FIT"]["mean_log2_rank"])
        - float(by_metric["DISCOUNTED_KT"]["mean_log2_rank"])
    ) / n
    oracle_gap = float(by_metric["DISCOUNTED_KT"]["mean_oracle_overhead_bits"]) / n
    payload = {
        "n": n,
        "rate": rate,
        "frames": frames,
        "switch_every": switch_every,
        "sessions": sessions,
        "discount": discount,
        "discounted_adaptive_gain_over_fixed_fit_per_symbol": adaptive_gain,
        "discounted_oracle_gap_per_symbol": oracle_gap,
        "pass": bool(adaptive_gain >= 0.03 and oracle_gap <= 0.08),
        "theory_warning": (
            "This empirical adaptation signal does not validate the proposal's signed-regret-to-average-error statement, "
            "which fails without one-sided control."
        ),
    }
    write_json(output_dir / "05_nonstationary_gate.json", payload)
    return payload
