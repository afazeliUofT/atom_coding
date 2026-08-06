from __future__ import annotations

import itertools
import math
from pathlib import Path
from typing import Any

import numpy as np

from .codelengths import ctw_log2_probability, kt_log2_probability, lz78_decode, lz78_records
from .code_geometry import LinearCode
from .individual_sequence import random_coding_formula_audit, regret_cancellation_counterexample
from .markov_types import enumerate_binary_markov_types, transition_counts
from .utils import binary_tuple, write_json


def run_exactness_audit(output_dir: Path, rng: np.random.Generator) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    # LZ78 fixed-block injectivity and round trip.
    seen_records: dict[tuple, tuple[int, ...]] = {}
    cases = 0
    collision = None
    for n in range(0, 11):
        for bits in itertools.product((0, 1), repeat=n):
            records = lz78_records(bits)
            decoded = lz78_decode(records, n)
            cases += 1
            if decoded != bits:
                collision = {"n": n, "bits": bits, "decoded": decoded}
                break
            key = (n, records)
            previous = seen_records.get(key)
            if previous is not None and previous != bits:
                collision = {"n": n, "bits": bits, "previous": previous}
                break
            seen_records[key] = bits
        if collision:
            break
    checks.append({"name": "lz78_fixed_block_injective_roundtrip", "pass": collision is None, "detail": {"cases": cases, "failure": collision}})

    # CTW and KT normalization over all binary blocks.
    normalization_rows = []
    normalization_pass = True
    for n in range(1, 9):
        blocks = list(itertools.product((0, 1), repeat=n))
        for depth in range(0, 5):
            total = sum(2.0 ** ctw_log2_probability(bits, depth=depth) for bits in blocks)
            normalization_rows.append({"model": f"CTW_D{depth}", "n": n, "sum": total})
            normalization_pass &= math.isclose(total, 1.0, rel_tol=0.0, abs_tol=2e-11)
        for order in (0, 1, 2):
            total = sum(2.0 ** kt_log2_probability(bits, order=order) for bits in blocks)
            normalization_rows.append({"model": f"KT_O{order}", "n": n, "sum": total})
            normalization_pass &= math.isclose(total, 1.0, rel_tol=0.0, abs_tol=2e-11)
    checks.append({"name": "ctw_kt_probability_normalization", "pass": normalization_pass, "detail": {"rows": normalization_rows}})

    # Markov type counts against exact enumeration.
    type_pass = True
    type_rows = []
    for n in range(1, 11):
        exact: dict[tuple[int, int, int, int, int], int] = {}
        for bits in itertools.product((0, 1), repeat=n):
            key = transition_counts(bits)
            exact[key] = exact.get(key, 0) + 1
        types = enumerate_binary_markov_types(n)
        if len(types) != len(exact):
            type_pass = False
        maximum_error = 0.0
        for item in types:
            observed = exact.get(item.key, 0)
            expected_log = math.log2(observed) if observed else float("-inf")
            maximum_error = max(maximum_error, abs(item.log2_count - expected_log))
        type_pass &= maximum_error <= 1e-10
        type_rows.append({"n": n, "types": len(types), "maximum_log2_count_error": maximum_error})
    checks.append({"name": "binary_markov_type_counting", "pass": type_pass, "detail": {"rows": type_rows}})

    formula = random_coding_formula_audit()
    checks.append({"name": "random_coding_hypergeometric_formula", "pass": formula["cases_checked"] > 0, "detail": formula})

    regret = regret_cancellation_counterexample()
    checks.append({"name": "signed_regret_counterexample_reproduced", "pass": regret["structural_failure"], "detail": regret})

    # Random systematic code closure and cardinality.
    code_pass = True
    code_rows = []
    for n, k in ((8, 4), (10, 7), (12, 6)):
        code = LinearCode.random_systematic(n, k, rng)
        words = set(int(v) for v in code.codewords)
        closed = all((a ^ b) in words for a in list(words)[: min(32, len(words))] for b in list(words)[: min(32, len(words))])
        passed = len(words) == 1 << k and 0 in words and closed
        code_pass &= passed
        code_rows.append({"n": n, "k": k, "cardinality": len(words), "closed_sample": closed})
    checks.append({"name": "random_systematic_linear_code", "pass": code_pass, "detail": {"rows": code_rows}})

    payload = {"pass": all(bool(row["pass"]) for row in checks), "count": len(checks), "checks": checks}
    write_json(output_dir / "01_exactness_audit.json", payload)
    return payload
