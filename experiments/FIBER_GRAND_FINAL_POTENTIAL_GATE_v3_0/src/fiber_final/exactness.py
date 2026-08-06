from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .channels import FixedDeletionBSC, sample_channel
from .codes import VTOracle, crc_code, random_linear
from .history_decoder import history_decode
from .likelihood import deletion_likelihood, exhaustive_ml
from .moment_theory import exact_moment_rates
from .physical_model import simulate_timing_slip_bits
from .prefix_astar import prefix_aggregate_astar
from .shell_theory import certificate_inequality_holds, shell_certificate_bound
from .syndrome_trellis import syndrome_trellis_aggregate_decode
from .utils import delete_positions, write_json
from .vt_linear import vt_decode_single_deletion_linear


def _general_one_deletion_likelihood(word: int, received: int, channel: FixedDeletionBSC) -> float:
    q = 1.0 / channel.n
    total = 0.0
    for j in range(channel.n):
        survivor = delete_positions(word, (j,), channel.n)
        d = (survivor ^ received).bit_count()
        if channel.p == 0.0:
            total += q if d == 0 else 0.0
        else:
            total += q * (channel.p**d) * ((1.0 - channel.p) ** (channel.m - d))
    return total


def run_exactness_audit(output_dir: Path, rng: np.random.Generator) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    # Fast one-deletion recurrence versus the direct alignment formula.
    max_error = 0.0
    cases = 0
    for n in (5, 7, 9):
        for p in (0.0, 0.02, 0.2):
            channel = FixedDeletionBSC(n, 1, p)
            for word in range(1 << n):
                for received in range(1 << (n - 1)):
                    fast, _ = deletion_likelihood(word, received, channel)
                    direct = _general_one_deletion_likelihood(word, received, channel)
                    max_error = max(max_error, abs(fast - direct))
                    cases += 1
                if n == 9 and word >= 31:
                    break
    checks.append(
        {
            "name": "one_deletion_linear_recurrence",
            "pass": max_error <= 5e-15,
            "detail": {"cases": cases, "maximum_absolute_error": max_error},
        }
    )

    # Code oracle closure.
    code_rows = []
    for code in (random_linear(10, 7, rng, "_AUDIT"), crc_code(10, 7, label="_AUDIT")):
        closed = all(code.is_codeword(code.encode(m))[0] for m in range(1 << code.k))
        code_rows.append({"family": code.family, "closed": bool(closed), "size": code.size})
    checks.append({"name": "code_oracle_closure", "pass": all(v["closed"] for v in code_rows), "detail": code_rows})

    # Three exact one-deletion decoders versus exhaustive ML, including full ties.
    code = random_linear(8, 5, rng, "_MICRO")
    codewords = code.enumerate_codewords()
    failures = []
    one_cases = 0
    for p in (0.0, 0.02, 0.05, 0.2):
        channel = FixedDeletionBSC(8, 1, p)
        for _ in range(50):
            word = code.sample_codeword(rng)
            received, _, _ = sample_channel(word, channel, rng)
            ties_idx, _, _ = exhaustive_ml(codewords, received, channel)
            exact_ties = {int(codewords[i]) for i in ties_idx}
            fiber = history_decode(received, channel, code, 100000)
            trellis = syndrome_trellis_aggregate_decode(received, channel, code, 100000)
            prefix = prefix_aggregate_astar(received, channel, code, 100000)
            one_cases += 1
            if not (
                fiber.certified
                and trellis.certified
                and prefix.certified
                and set(fiber.tie_words) == exact_ties
                and set(trellis.tie_words) == exact_ties
                and set(prefix.tie_words) == exact_ties
            ):
                failures.append(
                    {
                        "p": p,
                        "received": received,
                        "exact_ties": sorted(exact_ties),
                        "fiber": fiber.to_dict(),
                        "trellis": trellis.to_dict(),
                        "prefix_ties": list(prefix.tie_words),
                    }
                )
                break
    checks.append(
        {
            "name": "one_deletion_complete_tie_exactness",
            "pass": not failures,
            "detail": {"cases": one_cases, "failure": failures[:1]},
        }
    )

    # Two-deletion exact history search.
    code2 = random_linear(8, 5, rng, "_D2_MICRO")
    codewords2 = code2.enumerate_codewords()
    two_failures = []
    two_cases = 0
    for p in (0.0, 0.02, 0.05):
        channel = FixedDeletionBSC(8, 2, p)
        for _ in range(35):
            word = code2.sample_codeword(rng)
            received, _, _ = sample_channel(word, channel, rng)
            ties_idx, _, _ = exhaustive_ml(codewords2, received, channel)
            exact_ties = {int(codewords2[i]) for i in ties_idx}
            fiber = history_decode(received, channel, code2, 250000)
            two_cases += 1
            if not fiber.certified or set(fiber.tie_words) != exact_ties:
                two_failures.append({"p": p, "received": received, "fiber": fiber.to_dict(), "ties": sorted(exact_ties)})
                break
    checks.append(
        {
            "name": "two_deletion_complete_tie_exactness",
            "pass": not two_failures,
            "detail": {"cases": two_cases, "failure": two_failures[:1]},
        }
    )

    # Exhaustive classical VT reconstruction through n=13.
    vt_failures = []
    vt_cases = 0
    for n in range(3, 14):
        vt = VTOracle(n, 0)
        for word in range(1 << n):
            if not vt.is_codeword(word)[0]:
                continue
            for deleted in range(n):
                received = delete_positions(word, (deleted,), n)
                outcome = vt_decode_single_deletion_linear(received, vt)
                vt_cases += 1
                if not outcome.valid or outcome.word != word:
                    vt_failures.append({"n": n, "word": word, "deleted": deleted, "decoded": outcome.word})
                    break
            if vt_failures:
                break
        if vt_failures:
            break
    checks.append(
        {
            "name": "linear_time_vt_exhaustive",
            "pass": not vt_failures,
            "detail": {"cases": vt_cases, "failure": vt_failures[:1]},
        }
    )

    # Shell-certificate inequality over all feasible error weights.
    shell_failures = []
    shell_cases = 0
    for n in (8, 12, 20, 32, 64):
        for t in (1, 2):
            for p in (0.0, 0.005, 0.02, 0.05, 0.1, 0.2):
                channel = FixedDeletionBSC(n, t, p)
                for e in range(channel.m + 1):
                    shell_cases += 1
                    if not certificate_inequality_holds(channel, e):
                        shell_failures.append({"n": n, "t": t, "p": p, "e": e, "bound": shell_certificate_bound(channel, e).to_dict()})
                        break
    checks.append(
        {
            "name": "deterministic_shell_certificate",
            "pass": not shell_failures,
            "detail": {"cases": shell_cases, "failure": shell_failures[:1]},
        }
    )

    # Moment rate must be sandwiched at finite n.
    moment_failures = []
    for n in (32, 64, 128):
        for t in (1, 2):
            for p in (0.01, 0.05):
                for rho in (0.5, 1.0, 2.0):
                    rates = exact_moment_rates(n, t, p, rho)
                    if rates.reveal_lower > rates.reveal_upper + 1e-12 or rates.certificate_upper < rates.reveal_upper - 1e-12:
                        moment_failures.append({"n": n, "t": t, "p": p, "rho": rho, "rates": rates.__dict__})
    checks.append(
        {
            "name": "moment_sandwich",
            "pass": not moment_failures,
            "detail": {"cases": 3 * 2 * 2 * 3, "failure": moment_failures[:1]},
        }
    )

    # Synthetic timing model creates exactly one deletion.
    timing_failures = []
    for _ in range(200):
        bits = tuple(int(v) for v in rng.integers(0, 2, size=32))
        _, deleted, errors = simulate_timing_slip_bits(bits, 7.0, 0.06, rng)
        if not (0 <= deleted < 32 and len(errors) == 31):
            timing_failures.append({"deleted": deleted, "error_length": len(errors)})
            break
    checks.append(
        {
            "name": "synthetic_timing_slip_structure",
            "pass": not timing_failures,
            "detail": {"cases": 200, "failure": timing_failures[:1]},
        }
    )

    payload = {"count": len(checks), "pass": all(bool(v["pass"]) for v in checks), "checks": checks}
    write_json(output_dir / "00_exactness_audit.json", payload)
    return payload
