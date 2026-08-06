from __future__ import annotations

import itertools
import math
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np

from .baselines import exhaustive_deletion_ml, exhaustive_insertion_ml
from .channels import DeletionChannel, InsertionChannel, sample_deletion_channel
from .codes import SystematicLinearCode, VTCode, crc_systematic_code, hamming_15_11_code
from .history_decoder import (
    deletion_history_model,
    history_fiber_decode,
    insertion_history_model,
)
from .likelihood import (
    deletion_likelihood_dp,
    deletion_likelihood_vectorized,
    one_deletion_likelihood,
    one_deletion_likelihood_fraction,
    one_insertion_likelihood,
)
from .prefix_decoder import prefix_fiber_decode
from .utils import binary_tuple, write_json


def _enumerative_one_deletion_score(word: int, received: int, n: int, p: float) -> float:
    x = binary_tuple(word, n)
    y = binary_tuple(received, n - 1)
    score = 0.0
    for deleted in range(n):
        survivors = x[:deleted] + x[deleted + 1 :]
        d = sum(int(a != b) for a, b in zip(survivors, y))
        score += (1.0 / n) * ((d == 0) if p == 0.0 else p**d * (1.0 - p) ** (n - 1 - d))
    return float(score)


def audit_likelihood_identities() -> dict[str, Any]:
    cases = 0
    maximum_error = 0.0
    maximum_vector_error = 0.0
    for n in range(3, 8):
        for p in (0.0, 0.1, 0.25):
            channel = DeletionChannel(n=n, deletions=1, substitution_probability=p)
            words = np.arange(1 << n, dtype=np.uint64)
            array = np.asarray([binary_tuple(int(v), n) for v in words], dtype=np.uint8)
            for received in range(1 << (n - 1)):
                vector, _ = deletion_likelihood_vectorized(array, received, channel)
                for word in range(1 << n):
                    direct = _enumerative_one_deletion_score(word, received, n, p)
                    recurrence = one_deletion_likelihood(word, received, channel)
                    dynamic = deletion_likelihood_dp(word, received, channel)
                    maximum_error = max(maximum_error, abs(direct - recurrence), abs(direct - dynamic))
                    maximum_vector_error = max(maximum_vector_error, abs(direct - float(vector[word])))
                    cases += 1
    return {
        "cases": cases,
        "maximum_scalar_error": maximum_error,
        "maximum_vector_error": maximum_vector_error,
        "pass": maximum_error <= 2e-12 and maximum_vector_error <= 2e-12,
    }


def audit_fraction_score() -> dict[str, Any]:
    n = 6
    p = Fraction(1, 5)
    maximum_error = Fraction(0, 1)
    for word in range(1 << n):
        for received in range(1 << (n - 1)):
            exact = one_deletion_likelihood_fraction(word, received, n, p)
            floating = Fraction.from_float(
                one_deletion_likelihood(word, received, DeletionChannel(n, 1, float(p)))
            ).limit_denominator(10**12)
            maximum_error = max(maximum_error, abs(exact - floating))
    return {
        "n": n,
        "p": str(p),
        "maximum_fraction_to_float_error": float(maximum_error),
        "pass": float(maximum_error) <= 1e-12,
    }


def audit_history_and_prefix_decoders(rng: np.random.Generator) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    exact_all = True
    false_certificates = 0
    for n, k in ((6, 3), (7, 4), (8, 5)):
        code = SystematicLinearCode.random_systematic(n, k, rng, name=f"AUDIT_RLC_{n}_{k}")
        for p in (0.0, 0.08, 0.2):
            channel = DeletionChannel(n=n, deletions=1, substitution_probability=p)
            for received in range(1 << (n - 1)):
                reference = exhaustive_deletion_ml(code, received, channel)
                history = history_fiber_decode(
                    deletion_history_model(received, channel),
                    code,
                    reference.tie_set,
                    reference.decision,
                    reference.work,
                    max_histories=1_000_000,
                )
                prefix_l0 = prefix_fiber_decode(
                    received,
                    channel,
                    code,
                    reference.tie_set,
                    reference.decision,
                    reference.work,
                    max_nodes=1_000_000,
                    use_prefix_feasibility=False,
                )
                prefix_l2 = prefix_fiber_decode(
                    received,
                    channel,
                    code,
                    reference.tie_set,
                    reference.decision,
                    reference.work,
                    max_nodes=1_000_000,
                    use_prefix_feasibility=True,
                )
                for label, result in (
                    ("history", history),
                    ("prefix_l0", prefix_l0),
                    ("prefix_l2", prefix_l2),
                ):
                    exact_all &= result.exact
                    if result.certified and not result.exact:
                        false_certificates += 1
                    rows.append(
                        {
                            "n": n,
                            "k": k,
                            "p": p,
                            "received": received,
                            "algorithm": label,
                            "exact": result.exact,
                            "certified": result.certified,
                            "fallback": result.fallback_used,
                        }
                    )
    return {
        "cases": len(rows),
        "exact_all": bool(exact_all),
        "false_certificates": false_certificates,
        "pass": bool(exact_all and false_certificates == 0),
    }


def audit_two_deletion_decoder(rng: np.random.Generator) -> dict[str, Any]:
    code = SystematicLinearCode.random_systematic(9, 6, rng, name="AUDIT_T2")
    channel = DeletionChannel(9, 2, 0.12)
    exact_all = True
    certified_count = 0
    for received in range(1 << 7):
        reference = exhaustive_deletion_ml(code, received, channel)
        result = history_fiber_decode(
            deletion_history_model(received, channel),
            code,
            reference.tie_set,
            reference.decision,
            reference.work,
            max_histories=2_000_000,
        )
        exact_all &= result.exact
        certified_count += int(result.certified)
    return {
        "received_cases": 1 << 7,
        "exact_all": bool(exact_all),
        "certified_count": certified_count,
        "pass": bool(exact_all and certified_count == (1 << 7)),
    }


def audit_insertion_decoder(rng: np.random.Generator) -> dict[str, Any]:
    code = SystematicLinearCode.random_systematic(7, 4, rng, name="AUDIT_I1")
    channel = InsertionChannel(7, 0.11)
    exact_all = True
    certified_count = 0
    for received in range(1 << 8):
        reference = exhaustive_insertion_ml(code, received, channel)
        result = history_fiber_decode(
            insertion_history_model(received, channel),
            code,
            reference.tie_set,
            reference.decision,
            reference.work,
            max_histories=2_000_000,
        )
        exact_all &= result.exact
        certified_count += int(result.certified)
    return {
        "received_cases": 1 << 8,
        "exact_all": bool(exact_all),
        "certified_count": certified_count,
        "pass": bool(exact_all and certified_count == (1 << 8)),
    }


def audit_code_families(rng: np.random.Generator) -> dict[str, Any]:
    codes = [
        SystematicLinearCode.random_systematic(12, 9, rng, "AUDIT_RLC"),
        crc_systematic_code(12, 8, 0b10011, "AUDIT_CRC"),
        hamming_15_11_code(),
        VTCode(11, 0),
    ]
    rows = []
    for code in codes:
        closed = all(code.message_index(int(word)) is not None for word in code.codewords_int)
        prefix_checks = []
        if isinstance(code, SystematicLinearCode):
            for length in range(code.n + 1):
                for message in range(min(code.size, 32)):
                    word = int(code.codewords_int[message])
                    prefix = word & ((1 << length) - 1) if length else 0
                    feasible, _ = code.prefix_feasible(prefix, length)
                    prefix_checks.append(feasible)
        rows.append(
            {
                "name": code.name,
                "family": code.family,
                "size": code.size,
                "closed": closed,
                "prefix_checks_pass": all(prefix_checks) if prefix_checks else True,
            }
        )
    return {"codes": rows, "pass": all(row["closed"] and row["prefix_checks_pass"] for row in rows)}


def run_exactness_audit(output_dir: Path, rng: np.random.Generator) -> dict[str, Any]:
    checks = [
        {"name": "likelihood_identities", "detail": audit_likelihood_identities()},
        {"name": "exact_fraction_score", "detail": audit_fraction_score()},
        {"name": "history_and_prefix_decoders", "detail": audit_history_and_prefix_decoders(rng)},
        {"name": "two_deletion_decoder", "detail": audit_two_deletion_decoder(rng)},
        {"name": "insertion_decoder", "detail": audit_insertion_decoder(rng)},
        {"name": "code_families", "detail": audit_code_families(rng)},
    ]
    for check in checks:
        check["pass"] = bool(check["detail"].get("pass", False))
    payload = {"checks": checks, "count": len(checks), "pass": all(check["pass"] for check in checks)}
    write_json(output_dir / "01_exactness_audit.json", payload)
    return payload
