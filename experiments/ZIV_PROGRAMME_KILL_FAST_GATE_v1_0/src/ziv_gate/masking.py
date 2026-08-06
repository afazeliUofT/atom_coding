from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd

from .codelengths import ctw_codelength_bits, lz78_codelength_bits
from .utils import binary_tuple, slope_linear, write_json


LengthFunction = Callable[[tuple[int, ...]], float]


def codelength_functions(ctw_depths: Sequence[int]) -> dict[str, LengthFunction]:
    functions: dict[str, LengthFunction] = {"LZ78_FIXED_BLOCK": lambda bits: float(lz78_codelength_bits(bits))}
    for depth in ctw_depths:
        functions[f"CTW_D{depth}"] = lambda bits, d=depth: float(ctw_codelength_bits(bits, depth=d))
    return functions


def _defects(lu: float, lv: float, lw: float) -> tuple[float, float, float]:
    return lw - lu - lv, lu - lv - lw, lv - lu - lw


def exact_masking_search(output_dir: Path, blocklengths: Sequence[int], ctw_depths: Sequence[int]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    functions = codelength_functions(ctw_depths)
    for n in blocklengths:
        ambient = 1 << n
        indices = np.arange(ambient, dtype=np.int64)
        sequences = [binary_tuple(value, n) for value in range(ambient)]
        for name, function in functions.items():
            lengths = np.array([function(bits) for bits in sequences], dtype=float)
            maxima = [float("-inf"), float("-inf"), float("-inf")]
            witnesses = [(0, 0), (0, 0), (0, 0)]
            proposal_max = float("-inf")
            proposal_witness = (0, 0)
            block = 256 if ambient >= 256 else ambient
            for start in range(0, ambient, block):
                stop = min(ambient, start + block)
                u = indices[start:stop, None]
                v = indices[None, :]
                x = np.bitwise_xor(u, v)
                lu = lengths[start:stop, None]
                lv = lengths[None, :]
                lw = lengths[x]
                matrices = (lw - lu - lv, lu - lv - lw, lv - lu - lw)
                for kind, matrix in enumerate(matrices):
                    flat = int(np.argmax(matrix))
                    value = float(matrix.flat[flat])
                    if value > maxima[kind]:
                        row, col = np.unravel_index(flat, matrix.shape)
                        maxima[kind] = value
                        witnesses[kind] = (start + int(row), int(col))
                proposal = lw - np.abs(lu - lv)
                flat = int(np.argmax(proposal))
                value = float(proposal.flat[flat])
                if value > proposal_max:
                    row, col = np.unravel_index(flat, proposal.shape)
                    proposal_max = value
                    proposal_witness = (start + int(row), int(col))
            row = {
                "metric": name,
                "n": n,
                "ambient": ambient,
                "max_upper_subadditivity_defect": maxima[0],
                "max_inverse_u_defect": maxima[1],
                "max_inverse_v_defect": maxima[2],
                "max_positive_defect": max(0.0, *maxima),
                "max_positive_defect_per_symbol": max(0.0, *maxima) / n,
                "upper_witness_u": witnesses[0][0],
                "upper_witness_v": witnesses[0][1],
                "inverse_u_witness_u": witnesses[1][0],
                "inverse_u_witness_v": witnesses[1][1],
                "proposal_original_delta_max": proposal_max,
                "proposal_delta_per_symbol": proposal_max / n,
                "proposal_witness_u": proposal_witness[0],
                "proposal_witness_v": proposal_witness[1],
            }
            rows.append(row)
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "03_masking_exact.csv", index=False)
    metric_summary: dict[str, Any] = {}
    for metric, group in frame.groupby("metric"):
        largest = group.loc[group["n"].idxmax()].to_dict()
        metric_summary[metric] = {
            "largest_n": int(largest["n"]),
            "largest_n_max_positive_defect": float(largest["max_positive_defect"]),
            "largest_n_normalized_defect": float(largest["max_positive_defect_per_symbol"]),
            "all_exact_lengths": [int(v) for v in group["n"].tolist()],
            "any_positive_violation": bool((group["max_positive_defect"] > 1e-10).any()),
            "original_probe_delta_linear_warning": bool(float(largest["proposal_delta_per_symbol"]) > 0.25),
        }
    return {"rows": rows, "metrics": metric_summary}


def _lfsr_sequence(n: int, seed: int = 0b11111) -> tuple[int, ...]:
    state = seed & 0b11111
    out: list[int] = []
    for _ in range(n):
        out.append(state & 1)
        feedback = ((state >> 0) ^ (state >> 2)) & 1
        state = (state >> 1) | (feedback << 4)
        if state == 0:
            state = 1
    return tuple(out)


def _thue_morse(n: int) -> tuple[int, ...]:
    return tuple((i.bit_count() & 1) for i in range(n))


def _markov_sequence(n: int, p01: float, p10: float, rng: np.random.Generator) -> tuple[int, ...]:
    pi1 = p01 / (p01 + p10)
    bit = int(rng.random() < pi1)
    out = [bit]
    for _ in range(n - 1):
        if bit == 0:
            bit = int(rng.random() < p01)
        else:
            bit = int(not (rng.random() < p10))
        out.append(bit)
    return tuple(out)


def structured_corpus(n: int, rng: np.random.Generator, random_count: int) -> list[tuple[str, tuple[int, ...]]]:
    corpus: list[tuple[str, tuple[int, ...]]] = []
    corpus.extend(
        [
            ("zeros", (0,) * n),
            ("ones", (1,) * n),
            ("alternating01", tuple(i & 1 for i in range(n))),
            ("alternating10", tuple(1 - (i & 1) for i in range(n))),
            ("thue_morse", _thue_morse(n)),
            ("lfsr5", _lfsr_sequence(n)),
        ]
    )
    for period in (3, 4, 5, 7, 8, 13):
        pattern = tuple(int(v) for v in rng.integers(0, 2, size=period))
        corpus.append((f"period_{period}", tuple(pattern[i % period] for i in range(n))))
    for fraction in (0.05, 0.1, 0.25, 0.5, 0.75):
        length = max(1, int(round(n * fraction)))
        start = max(0, (n - length) // 2)
        bits = [0] * n
        bits[start : start + length] = [1] * length
        corpus.append((f"single_burst_{fraction:.2f}", tuple(bits)))
    for p01, p10, label in (
        (0.01, 0.20, "markov_rare_bursty"),
        (0.05, 0.05, "markov_balanced_sticky"),
        (0.08, 0.40, "markov_moderate"),
        (0.08, 0.92, "iid_008_control"),
    ):
        corpus.append((label, _markov_sequence(n, p01, p10, rng)))
    for i in range(random_count):
        p = float(rng.choice([0.05, 0.1, 0.25, 0.5]))
        corpus.append((f"bernoulli_{p:.2f}_{i}", tuple(int(v) for v in (rng.random(n) < p))))
    # Deduplicate by sequence while keeping the first label.
    seen: set[tuple[int, ...]] = set()
    output: list[tuple[str, tuple[int, ...]]] = []
    for label, bits in corpus:
        if bits not in seen:
            seen.add(bits)
            output.append((label, bits))
    return output


def heuristic_masking_search(
    output_dir: Path,
    rng: np.random.Generator,
    blocklengths: Sequence[int],
    ctw_depths: Sequence[int],
    random_count: int,
    random_pairs: int,
    hill_steps: int,
) -> dict[str, Any]:
    functions = codelength_functions(ctw_depths)
    rows: list[dict[str, Any]] = []
    witness_rows: list[dict[str, Any]] = []
    for n in blocklengths:
        corpus = structured_corpus(n, rng, random_count)
        sequences = [bits for _, bits in corpus]
        labels = [label for label, _ in corpus]
        pairs: list[tuple[int, int]] = [(i, j) for i in range(len(corpus)) for j in range(len(corpus))]
        for _ in range(random_pairs):
            pairs.append((int(rng.integers(len(corpus))), int(rng.integers(len(corpus)))))
        for metric, function in functions.items():
            if metric.startswith("CTW") and n > 128:
                continue
            cache: dict[tuple[int, ...], float] = {}
            def length(bits: tuple[int, ...]) -> float:
                value = cache.get(bits)
                if value is None:
                    value = function(bits)
                    cache[bits] = value
                return value
            best = [float("-inf"), float("-inf"), float("-inf")]
            witnesses: list[tuple[tuple[int, ...], tuple[int, ...], str, str]] = [
                (sequences[0], sequences[0], labels[0], labels[0]) for _ in range(3)
            ]
            proposal_max = float("-inf")
            for i, j in pairs:
                u, v = sequences[i], sequences[j]
                w = tuple(a ^ b for a, b in zip(u, v, strict=True))
                lu, lv, lw = length(u), length(v), length(w)
                values = _defects(lu, lv, lw)
                for k, value in enumerate(values):
                    if value > best[k]:
                        best[k] = value
                        witnesses[k] = (u, v, labels[i], labels[j])
                proposal_max = max(proposal_max, lw - abs(lu - lv))
            # Stochastic hill climbing targets the upper defect. It is applied to
            # LZ78 only; exact enumeration already supplies CTW counterexamples and
            # repeated CTW tree rebuilds would dominate the kill-fast runtime.
            hill_restarts = min(8, max(1, random_count // 2)) if metric == "LZ78_FIXED_BLOCK" else 0
            for restart in range(hill_restarts):
                u = list(sequences[int(rng.integers(len(sequences)))])
                v = list(sequences[int(rng.integers(len(sequences)))])
                def objective(aa: list[int], bb: list[int]) -> float:
                    tu, tv = tuple(aa), tuple(bb)
                    tw = tuple(x ^ y for x, y in zip(tu, tv, strict=True))
                    return length(tw) - length(tu) - length(tv)
                current = objective(u, v)
                for _ in range(hill_steps):
                    target = u if rng.random() < 0.5 else v
                    position = int(rng.integers(n))
                    target[position] ^= 1
                    candidate = objective(u, v)
                    if candidate >= current or rng.random() < 0.01:
                        current = candidate
                    else:
                        target[position] ^= 1
                    if current > best[0]:
                        best[0] = current
                        witnesses[0] = (tuple(u), tuple(v), f"hill_u_{restart}", f"hill_v_{restart}")
            row = {
                "metric": metric,
                "n": n,
                "corpus_size": len(corpus),
                "evaluated_pair_count": len(pairs),
                "max_upper_subadditivity_defect": best[0],
                "max_inverse_u_defect": best[1],
                "max_inverse_v_defect": best[2],
                "max_positive_defect": max(0.0, *best),
                "max_positive_defect_per_symbol": max(0.0, *best) / n,
                "proposal_original_delta_max": proposal_max,
                "proposal_delta_per_symbol": proposal_max / n,
            }
            rows.append(row)
            for kind, (u, v, label_u, label_v) in zip(("upper", "inverse_u", "inverse_v"), witnesses, strict=True):
                witness_rows.append(
                    {
                        "metric": metric,
                        "n": n,
                        "kind": kind,
                        "value": best[("upper", "inverse_u", "inverse_v").index(kind)],
                        "label_u": label_u,
                        "label_v": label_v,
                        "u_prefix": "".join(str(x) for x in u[:128]),
                        "v_prefix": "".join(str(x) for x in v[:128]),
                    }
                )
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "03_masking_heuristic.csv", index=False)
    pd.DataFrame(witness_rows).to_csv(output_dir / "03_masking_witnesses.csv", index=False)
    metric_summary: dict[str, Any] = {}
    for metric, group in frame.groupby("metric"):
        ordered = group.sort_values("n")
        largest = ordered.iloc[-1]
        metric_summary[metric] = {
            "largest_n": int(largest["n"]),
            "largest_n_normalized_positive_defect": float(largest["max_positive_defect_per_symbol"]),
            "any_positive_violation": bool((ordered["max_positive_defect"] > 1e-10).any()),
            "normalized_defect_linear_slope": slope_linear(ordered["n"], ordered["max_positive_defect"]),
            "original_probe_delta_per_symbol_at_largest_n": float(largest["proposal_delta_per_symbol"]),
        }
    return {"rows": rows, "metrics": metric_summary}


def run_masking_gate(
    output_dir: Path,
    rng: np.random.Generator,
    exact_blocklengths: Sequence[int],
    heuristic_blocklengths: Sequence[int],
    ctw_depths: Sequence[int],
    random_count: int,
    random_pairs: int,
    hill_steps: int,
) -> dict[str, Any]:
    exact = exact_masking_search(output_dir, exact_blocklengths, ctw_depths)
    heuristic = heuristic_masking_search(
        output_dir,
        rng,
        heuristic_blocklengths,
        ctw_depths,
        random_count,
        random_pairs,
        hill_steps,
    )
    combined: dict[str, Any] = {}
    for metric in sorted(set(exact["metrics"]) | set(heuristic["metrics"])):
        e = exact["metrics"].get(metric, {})
        h = heuristic["metrics"].get(metric, {})
        combined[metric] = {
            "exact_any_positive_violation": bool(e.get("any_positive_violation", False)),
            "heuristic_any_positive_violation": bool(h.get("any_positive_violation", False)),
            "largest_tested_n": h.get("largest_n", e.get("largest_n")),
            "largest_normalized_positive_defect": h.get("largest_n_normalized_positive_defect", e.get("largest_n_normalized_defect", 0.0)),
            "survives_falsification_only": not bool(e.get("any_positive_violation", False) or h.get("any_positive_violation", False)),
        }
    payload = {
        "exact": exact,
        "heuristic": heuristic,
        "combined": combined,
        "probe_definition_audit": {
            "proposal_metric_is_not_a_defect": True,
            "correct_upper_defect": "[L(u xor v)-L(u)-L(v)]_+",
            "correct_inverse_defects": "[L(u)-L(v)-L(u xor v)]_+ and the symmetric expression",
            "reason": (
                "The proposal's Delta=L(u xor v)-|L(u)-L(v)| can be linear for ordinary incompressible pairs "
                "even when neither required triangle inequality is violated."
            ),
        },
        "interpretation": (
            "No finite computation can prove a uniform o(n) masking theorem. A positive linear witness kills the chosen "
            "practical codelength; absence of a witness only authorizes theorem development under a frozen codelength."
        ),
    }
    write_json(output_dir / "03_masking_gate.json", payload)
    return payload
