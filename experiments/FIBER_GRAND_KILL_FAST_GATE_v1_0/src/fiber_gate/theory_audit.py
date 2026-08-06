from __future__ import annotations

import itertools
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .channels import DeletionChannel
from .codes import SystematicLinearCode
from .likelihood import deletion_likelihood_dp, exact_ml_indices
from .utils import binary_tuple, write_json


def fiber_guesswork_identity(max_n: int = 8) -> dict[str, Any]:
    """Verify that 'fiber guesswork' is ordinary posterior conditional guesswork.

    Under a uniform prior on X, posterior ordering P(X=x|Y=y) is exactly the
    ordering of W(y|x), so the proposal's G_fib is the classical optimal
    guessing rank of X with side information Y.
    """
    cases = 0
    maximum_order_disagreement = 0
    for n in range(3, max_n + 1):
        channel = DeletionChannel(n=n, deletions=1, substitution_probability=0.2)
        for received in range(1 << (n - 1)):
            scores = np.asarray(
                [deletion_likelihood_dp(word, received, channel) for word in range(1 << n)],
                dtype=float,
            )
            posterior = scores / max(float(np.sum(scores)), 1e-300)
            order_channel = np.argsort(-scores, kind="stable")
            order_posterior = np.argsort(-posterior, kind="stable")
            maximum_order_disagreement = max(
                maximum_order_disagreement,
                int(np.count_nonzero(order_channel != order_posterior)),
            )
            cases += 1
    return {
        "cases": cases,
        "maximum_order_disagreement": maximum_order_disagreement,
        "identity": "G_fib(X|Y) equals optimal posterior conditional guesswork G(X|Y) under a uniform input prior.",
        "novelty_consequence": (
            "The random variable and its generic Renyi/guessing bounds are not a new information-theoretic object. "
            "Novelty must come from channel-specific computation, search inflation, tractability, or decoder-tail theorems."
        ),
        "pass": maximum_order_disagreement == 0,
    }


def random_code_rank_identity(max_n: int = 6) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    maximum_error = 0.0
    for n in range(2, max_n + 1):
        universe = 1 << n
        for size in range(2, min(8, universe) + 1):
            for rank in range(1, universe + 1):
                bad = rank - 1
                total = math.comb(universe - 1, size - 1)
                good_pool = universe - rank
                exact = 0.0 if good_pool < size - 1 else math.comb(good_pool, size - 1) / total
                # Exhaustively enumerate representative small cases.
                if universe <= 16:
                    competitors = list(range(1, universe))
                    bad_set = set(competitors[:bad])
                    successes = 0
                    count = 0
                    for subset in itertools.combinations(competitors, size - 1):
                        count += 1
                        successes += int(not any(item in bad_set for item in subset))
                    observed = successes / count
                    maximum_error = max(maximum_error, abs(observed - exact))
                rows.append({"n": n, "N": universe, "M": size, "rank": rank, "success": exact})
    return {
        "cases": len(rows),
        "maximum_exhaustive_error": maximum_error,
        "formula": "C(N-g, M-1) / C(N-1, M-1)",
        "interpretation": "Correct finite-block combinatorial identity, but elementary once posterior rank is defined.",
        "pass": maximum_error <= 1e-15,
    }


def multiplicity_counterexample() -> dict[str, Any]:
    # C={000,001}, y=00, exactly one uniform deletion, no substitutions.
    channel = DeletionChannel(n=3, deletions=1, substitution_probability=0.0)
    candidates = [0b000, 0b001]
    scores = [deletion_likelihood_dp(word, 0b00, channel) for word in candidates]
    return {
        "codewords": ["000", "001"],
        "received": "00",
        "scores": {"000": scores[0], "001": scores[1]},
        "best_path_tie": True,
        "aggregate_unique_ml": "000",
        "pass": bool(abs(scores[0] - 1.0) < 1e-15 and abs(scores[1] - 1.0 / 3.0) < 1e-15),
    }


def run_theory_audit(output_dir: Path) -> dict[str, Any]:
    payload = {
        "fiber_guesswork_identity": fiber_guesswork_identity(),
        "random_code_rank_identity": random_code_rank_identity(),
        "multiplicity_counterexample": multiplicity_counterexample(),
        "manual_novelty_status": "PENDING_PRIMARY_SOURCE_ADJUDICATION",
    }
    payload["pass"] = bool(
        payload["fiber_guesswork_identity"]["pass"]
        and payload["random_code_rank_identity"]["pass"]
        and payload["multiplicity_counterexample"]["pass"]
    )
    write_json(output_dir / "02_theory_boundary.json", payload)
    return payload
