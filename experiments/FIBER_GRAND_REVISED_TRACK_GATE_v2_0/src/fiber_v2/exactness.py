from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from .channels import FixedDeletionBSC, sample_channel
from .codes import VTOracle, crc_like, random_linear
from .history_decoder import history_decode
from .likelihood import exhaustive_ml
from .shell_theory import certificate_inequality_holds, shell_certificate_bound
from .prefix_astar import prefix_aggregate_astar
from .syndrome_trellis import syndrome_trellis_aggregate_decode
from .utils import write_json
from .vt_baseline import vt_direct_one_deletion


def run_exactness_audit(output_dir: Path, rng: np.random.Generator) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    # Linear syndrome and encoding closure without a codebook dictionary.
    linear_rows = []
    for family in ("RLC", "CRC"):
        code = random_linear(10, 7, rng, "_AUDIT") if family == "RLC" else crc_like(10, 7, label="_AUDIT")
        closed = True
        for message in range(1 << code.k):
            word = code.encode(message)
            valid, _ = code.is_codeword(word)
            closed &= valid
        linear_rows.append({"family": code.family, "closed": bool(closed), "size": code.size})
    checks.append({"name": "code_oracle_closure", "pass": all(row["closed"] for row in linear_rows), "detail": linear_rows})

    # One-deletion: two independent certified algorithms against exhaustive ML.
    code = random_linear(8, 5, rng, "_MICRO")
    codewords = code.enumerate_codewords()
    one_cases = 0
    one_failures = []
    for p in (0.0, 0.05, 0.2):
        channel = FixedDeletionBSC(8, 1, p)
        for _ in range(40):
            word = code.sample_codeword(rng)
            received, _, _ = sample_channel(word, channel, rng)
            ties_idx, _, _ = exhaustive_ml(codewords, received, channel)
            tie_words = {int(codewords[i]) for i in ties_idx}
            fiber = history_decode(received, channel, code, max_histories=100000)
            trellis = syndrome_trellis_aggregate_decode(received, channel, code, max_terminals=100000)
            prefix = prefix_aggregate_astar(received, channel, code, max_nodes=100000)
            one_cases += 1
            if (not fiber.certified or not trellis.certified or not prefix.certified or fiber.decision_word not in tie_words or trellis.decision_word not in tie_words or prefix.decision_word not in tie_words):
                one_failures.append({"p": p, "received": received, "fiber": fiber.to_dict(), "trellis": trellis.to_dict(), "prefix_ties": list(prefix.tie_words), "ties": sorted(tie_words)})
                break
    checks.append({"name": "one_deletion_exact_decoders", "pass": not one_failures, "detail": {"cases": one_cases, "failure": one_failures[:1]}})

    # Two-deletion history decoder against exhaustive ML.
    code2 = random_linear(8, 5, rng, "_D2_MICRO")
    codewords2 = code2.enumerate_codewords()
    two_cases = 0
    two_failures = []
    for p in (0.0, 0.05):
        channel = FixedDeletionBSC(8, 2, p)
        for _ in range(30):
            word = code2.sample_codeword(rng)
            received, _, _ = sample_channel(word, channel, rng)
            ties_idx, _, _ = exhaustive_ml(codewords2, received, channel)
            tie_words = {int(codewords2[i]) for i in ties_idx}
            fiber = history_decode(received, channel, code2, max_histories=200000)
            two_cases += 1
            if not fiber.certified or fiber.decision_word not in tie_words:
                two_failures.append({"p": p, "received": received, "fiber": fiber.to_dict(), "ties": sorted(tie_words)})
                break
    checks.append({"name": "two_deletion_exact_history", "pass": not two_failures, "detail": {"cases": two_cases, "failure": two_failures[:1]}})

    # VT direct specialized decoder agrees with membership-only FIBER at p=0.
    vt = VTOracle(11, 0)
    channel_vt = FixedDeletionBSC(11, 1, 0.0)
    vt_cases = 0
    vt_failures = []
    for _ in range(60):
        word = vt.sample_codeword(rng)
        received, _, _ = sample_channel(word, channel_vt, rng)
        fiber = history_decode(received, channel_vt, vt, max_histories=10000)
        direct = vt_direct_one_deletion(received, vt)
        vt_cases += 1
        if not fiber.certified or set(fiber.tie_words) != set(direct.tie_words):
            vt_failures.append({"received": received, "fiber": fiber.to_dict(), "direct_ties": list(direct.tie_words)})
            break
    checks.append({"name": "vt_specialized_agreement", "pass": not vt_failures, "detail": {"cases": vt_cases, "failure": vt_failures[:1]}})

    # Deterministic shell certificate inequality for every feasible error count.
    shell_cases = 0
    shell_failures = []
    for n in (8, 12, 20, 32):
        for t in (1, 2):
            if t >= n:
                continue
            for p in (0.0, 0.02, 0.05, 0.1, 0.2):
                channel = FixedDeletionBSC(n, t, p)
                for e in range(channel.m + 1):
                    shell_cases += 1
                    if not certificate_inequality_holds(channel, e):
                        shell_failures.append({"n": n, "t": t, "p": p, "e": e, "bound": shell_certificate_bound(channel, e).to_dict()})
                        break
    checks.append({"name": "shell_certificate_theorem_numeric_audit", "pass": not shell_failures, "detail": {"cases": shell_cases, "failure": shell_failures[:1]}})

    payload = {"count": len(checks), "pass": all(bool(check["pass"]) for check in checks), "checks": checks}
    write_json(output_dir / "01_exactness_audit.json", payload)
    return payload
