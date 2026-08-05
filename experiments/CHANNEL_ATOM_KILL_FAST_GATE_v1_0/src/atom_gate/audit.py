from __future__ import annotations

import itertools
from pathlib import Path
from typing import Callable

import numpy as np

from .channels import (
    bac,
    bsc,
    is_cyclic_circulant_up_to_relabeling,
    noncyclic_reversible_action_channel,
)
from .codes import BinaryLinearCode
from .decoder import one_shot_residual_decode, strongest_binary_exact_reference
from .enumeration import ProductAtomEnumerator
from .metrics import expected_random_code_fiber, injective_mass, representation_kappa
from .models import DeterministicMap, Representation
from .representations import (
    bac_representations,
    compact_max_injective_mass,
    independent_row_coupling,
    max_injective_representation,
)
from .utils import write_json


def run_exactness_audit(output_dir: Path, rng: np.random.Generator) -> dict[str, object]:
    checks: list[dict[str, object]] = []

    def check(name: str, fn: Callable[[], object]) -> None:
        try:
            detail = fn()
            checks.append({"name": name, "pass": True, "detail": detail})
        except Exception as exc:
            checks.append({"name": name, "pass": False, "error": repr(exc)})

    def duplicate_reduction() -> dict[str, object]:
        maps = [DeterministicMap((0, 1)), DeterministicMap((0, 1)), DeterministicMap((1, 0))]
        rep = Representation("split", maps, np.array([0.3, 0.4, 0.3]))
        reduced = rep.reduced()
        if reduced.support_size != 2:
            raise AssertionError("Duplicate maps were not merged")
        if not np.allclose(rep.induced_channel(2), reduced.induced_channel(2)):
            raise AssertionError("Reduction changed the induced channel")
        return {"before": rep.support_size, "after": reduced.support_size}

    check("duplicate_map_reduction", duplicate_reduction)

    def bac_interval_exactness() -> dict[str, object]:
        a, b = 0.23, 0.37
        spec = bac(a, b)
        reps = bac_representations(a, b, grid_points=13)
        for rep in reps:
            rep.verify(spec.matrix)
            for y in range(2):
                result = one_shot_residual_decode(spec.matrix, rep.sorted_by_weight(), y)
                if not result.exact:
                    raise AssertionError(f"One-shot mismatch for {rep.name}, y={y}")
        return {"representations": len(reps), "received_symbols_checked": 2 * len(reps)}

    check("bac_complete_interval_exactness", bac_interval_exactness)

    def product_enumerator_order() -> dict[str, object]:
        probs = np.array([0.55, 0.30, 0.15])
        n = 4
        expected = sorted(
            [
                (tuple(state), float(np.prod([probs[i] for i in state])))
                for state in itertools.product(range(len(probs)), repeat=n)
            ],
            key=lambda item: (-item[1], item[0]),
        )
        enumerator = ProductAtomEnumerator(probs, n)
        actual = [(item.atom_indices, item.probability) for item in enumerator]
        if len(actual) != len(expected):
            raise AssertionError("Product enumerator missed states")
        expected_probs = sorted([p for _, p in expected], reverse=True)
        actual_probs = [p for _, p in actual]
        if not np.allclose(actual_probs, expected_probs, atol=1e-14):
            raise AssertionError("Product enumerator probability order/content mismatch")
        if len(set(state for state, _ in actual)) != len(actual):
            raise AssertionError("Product enumerator produced duplicates")
        return {"states": len(actual), "probability_sum": float(sum(actual_probs))}

    check("product_atom_enumerator", product_enumerator_order)

    def gf2_fiber_oracle() -> dict[str, object]:
        code = BinaryLinearCode.random_systematic(8, 4, rng, "audit_code")
        cases = 0
        for fixed_mask in range(1 << code.n):
            if fixed_mask.bit_count() > 4:
                continue
            for fixed_value in range(1 << code.n):
                if fixed_value & ~fixed_mask:
                    continue
                sol = code.fiber(fixed_mask, fixed_value)
                brute = np.array(
                    [
                        i
                        for i, word in enumerate(code.codewords_int)
                        if ((int(word) ^ fixed_value) & fixed_mask) == 0
                    ],
                    dtype=np.int64,
                )
                if not np.array_equal(np.sort(sol.message_indices), brute):
                    raise AssertionError(
                        f"GF2 fiber mismatch for mask={fixed_mask:#x}, value={fixed_value:#x}"
                    )
                cases += 1
                if cases >= 500:
                    return {"cases": cases, "code": code.to_dict()}
        return {"cases": cases, "code": code.to_dict()}

    check("gf2_coordinate_fiber_oracle", gf2_fiber_oracle)

    def syndrome_trellis_exactness() -> dict[str, object]:
        code = BinaryLinearCode.random_systematic(9, 6, rng, "trellis_audit_code")
        channels = [bsc(0.13).matrix, bac(0.17, 0.31).matrix]
        cases = 0
        references = {"direct_codeword_ml": 0, "syndrome_trellis_ml": 0}
        for channel in channels:
            for received_int in range(1 << code.n):
                received = tuple((received_int >> i) & 1 for i in range(code.n))
                result = strongest_binary_exact_reference(channel, code, received)
                references[str(result["selected_name"])] += 1
                cases += 1
                if cases >= 256:
                    return {
                        "cases": cases,
                        "reference_selection_counts": references,
                        "parity_check": code.parity_check.tolist(),
                    }
        return {"cases": cases, "reference_selection_counts": references}

    check("syndrome_trellis_vs_direct_ml", syndrome_trellis_exactness)

    def injective_lp_agreement() -> dict[str, object]:
        channels = [
            np.array([[0.7, 0.3], [0.4, 0.6]]),
            np.array([[0.75, 0.05, 0.20], [0.10, 0.65, 0.25]]),
            np.array([[0.40, 0.05, 0.55], [0.10, 0.25, 0.65]]),
        ]
        details = []
        for channel in channels:
            rep = max_injective_representation(channel)
            full = injective_mass(rep)
            compact = compact_max_injective_mass(channel)
            if not np.isclose(full, compact, atol=1e-8):
                raise AssertionError(f"Injective LP mismatch: full={full}, compact={compact}")
            details.append({"full": full, "compact": compact})
        return {"channels": details}

    check("injective_lp_full_vs_compact", injective_lp_agreement)

    def random_code_fiber_identity() -> dict[str, object]:
        rep = independent_row_coupling(np.array([[0.7, 0.3], [0.4, 0.6]]))
        kappa = representation_kappa(rep, 2)
        n = 3
        ambient = 1 << n
        code_size = 4
        total = 0.0
        count = 0
        # Exact average over transmitted word, product atom, and all choices of other codewords.
        product_atoms = list(itertools.product(range(rep.support_size), repeat=n))
        for x in range(ambient):
            remaining = [v for v in range(ambient) if v != x]
            for others in itertools.combinations(remaining, code_size - 1):
                codebook = {x, *others}
                for atom_indices in product_atoms:
                    weight = float(np.prod([rep.weights[i] for i in atom_indices]))
                    y_bits = []
                    for coordinate, atom_index in enumerate(atom_indices):
                        bit = (x >> coordinate) & 1
                        y_bits.append(rep.maps[atom_index](bit))
                    fiber = 0
                    for candidate in codebook:
                        matches = True
                        for coordinate, atom_index in enumerate(atom_indices):
                            bit = (candidate >> coordinate) & 1
                            if rep.maps[atom_index](bit) != y_bits[coordinate]:
                                matches = False
                                break
                        fiber += int(matches)
                    total += weight * fiber
                count += 1
        exact = total / count
        formula = expected_random_code_fiber(2, n, code_size, kappa)
        if not np.isclose(exact, formula, atol=1e-10):
            raise AssertionError(f"Random-code fiber identity mismatch: exact={exact}, formula={formula}")
        return {"exact": exact, "formula": formula, "kappa": kappa, "averaged_codebooks": count}

    check("random_code_fiber_theorem_tiny_exact", random_code_fiber_identity)

    def reversible_action_noncyclic() -> dict[str, object]:
        spec, rep = noncyclic_reversible_action_channel()
        rep.verify(spec.matrix)
        if is_cyclic_circulant_up_to_relabeling(spec.matrix):
            raise AssertionError("Declared noncyclic action channel is cyclic-circulant")
        outputs_per_input = [[atom(x) for atom in rep.maps] for x in range(rep.input_size)]
        if any(len(values) != len(set(values)) for values in outputs_per_input):
            raise AssertionError("Action atoms are not transition-unique")
        return {"matrix": spec.matrix.tolist(), "maps": [list(m.outputs) for m in rep.maps]}

    check("noncyclic_reversible_action_control", reversible_action_noncyclic)

    passed = all(bool(item["pass"]) for item in checks)
    summary = {"pass": passed, "checks": checks, "count": len(checks)}
    write_json(output_dir / "01_exactness_audit.json", summary)
    return summary
