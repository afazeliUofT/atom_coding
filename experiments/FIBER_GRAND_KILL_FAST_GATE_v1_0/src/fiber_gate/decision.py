from __future__ import annotations

from pathlib import Path
from typing import Any

from .contract import decision_contract
from .utils import write_json


CONTINUE = "CONTINUE_FIELD_DEFINING_FIBER_TRACK"
PIVOT_REVISED = "PIVOT_TO_REVISED_FIBER_TRACK"
PIVOT_NARROW = "PIVOT_TO_NARROW_ONE_EDIT_FIBER_TRACK"
STOP = "STOP_FIELD_DEFINING_FIBER_PROGRAMME"


def _meets(row: dict[str, Any], thresholds: dict[str, Any], slope: float) -> bool:
    return bool(
        row.get("exact_fraction", 0.0) >= 1.0
        and row.get("completion_fraction", 0.0) >= float(thresholds["minimum_completion_fraction"])
        and row.get("mean_speedup", 0.0) >= float(thresholds["minimum_mean_total_work_speedup"])
        and row.get("p05_speedup", 0.0) >= float(thresholds["minimum_p05_total_work_speedup"])
        and row.get("median_candidate_score_speedup", 0.0)
        >= float(thresholds["minimum_median_candidate_score_speedup"])
        and row.get("median_membership_fraction", 1.0)
        <= float(thresholds["maximum_median_membership_fraction"])
        and row.get("p99_membership_fraction", 1.0)
        <= float(thresholds["maximum_p99_membership_fraction"])
        and slope >= float(thresholds["minimum_log2_work_slope_advantage"])
    )


def evaluate_boundary(boundary: dict[str, Any]) -> dict[str, Any]:
    thresholds = decision_contract()["boundary"]
    results = list(boundary.get("results", {}).values())
    disagreement_configs = sum(
        row.get("pathwise_ml_disagreement_fraction", 0.0)
        >= float(thresholds["minimum_disagreement_fraction"])
        for row in results
    )
    max_median_inflation = max(
        (row.get("median_search_inflation_optimistic", float("inf")) for row in results),
        default=float("inf"),
    )
    max_history_per_candidate = max(
        (row.get("p95_history_per_candidate", float("inf")) for row in results),
        default=float("inf"),
    )
    passed = bool(
        boundary.get("exact_all", False)
        and disagreement_configs >= int(thresholds["minimum_configs_with_pathwise_ml_disagreement"])
        and max_median_inflation <= float(thresholds["maximum_median_search_inflation_optimistic"])
        and max_history_per_candidate <= float(thresholds["maximum_p95_history_per_candidate"])
    )
    return {
        "pass": passed,
        "disagreement_configs": disagreement_configs,
        "max_median_search_inflation": max_median_inflation,
        "max_p95_history_per_candidate": max_history_per_candidate,
    }


def evaluate_one_deletion(payload: dict[str, Any]) -> dict[str, Any]:
    thresholds = decision_contract()["one_deletion"]
    required_p = [float(v) for v in thresholds["required_probabilities"]]
    eligible = set(thresholds["eligible_algorithms"])
    reps = list(payload.get("representations", {}).values())
    families = sorted(set(str(row["code_family"]) for row in reps))
    family_details: dict[str, Any] = {}
    passing_families = 0
    for family in families:
        p_details: dict[str, Any] = {}
        family_pass = True
        for p in required_p:
            candidates = [
                row
                for row in reps
                if row["code_family"] == family
                and abs(float(row["p"]) - p) <= 1e-12
                and row["algorithm"] in eligible
            ]
            evaluated = []
            for candidate in candidates:
                largest = candidate["largest_n"]
                passed = _meets(largest, thresholds, float(candidate["slope_advantage"]))
                evaluated.append(
                    {
                        "algorithm": candidate["algorithm"],
                        "pass": passed,
                        "largest_n": largest,
                        "slope_advantage": candidate["slope_advantage"],
                    }
                )
            p_pass = any(item["pass"] for item in evaluated)
            family_pass &= p_pass
            p_details[str(p)] = {"pass": p_pass, "algorithms": evaluated}
        passing_families += int(family_pass)
        family_details[family] = {"pass": family_pass, "probabilities": p_details}
    passed = bool(
        payload.get("exact_all", False)
        and passing_families >= int(thresholds["minimum_passing_code_families"])
    )
    return {
        "pass": passed,
        "passing_families": passing_families,
        "family_details": family_details,
    }


def _evaluate_single_algorithm_family(
    payload: dict[str, Any],
    contract_key: str,
) -> dict[str, Any]:
    thresholds = decision_contract()[contract_key]
    required_p = [float(v) for v in thresholds["required_probabilities"]]
    reps = list(payload.get("representations", {}).values())
    families = sorted(set(str(row["code_family"]) for row in reps))
    details: dict[str, Any] = {}
    passing = 0
    for family in families:
        family_pass = True
        probs = {}
        for p in required_p:
            candidates = [
                row for row in reps if row["code_family"] == family and abs(float(row["p"]) - p) <= 1e-12
            ]
            if not candidates:
                p_pass = False
                chosen = None
            else:
                chosen = max(candidates, key=lambda row: row["largest_n"].get("mean_speedup", 0.0))
                p_pass = _meets(chosen["largest_n"], thresholds, float(chosen["slope_advantage"]))
            family_pass &= p_pass
            probs[str(p)] = {
                "pass": p_pass,
                "largest_n": None if chosen is None else chosen["largest_n"],
                "slope_advantage": None if chosen is None else chosen["slope_advantage"],
            }
        passing += int(family_pass)
        details[family] = {"pass": family_pass, "probabilities": probs}
    passed = bool(
        payload.get("exact_all", False)
        and passing >= int(thresholds["minimum_passing_code_families"])
    )
    return {"pass": passed, "passing_families": passing, "family_details": details}


def evaluate_code_transfer(payload: dict[str, Any]) -> dict[str, Any]:
    thresholds = decision_contract()["code_transfer"]
    required_p = float(thresholds["required_probability"])
    rows = list(payload.get("results", {}).values())
    families = sorted(set(str(row["code_family"]) for row in rows))
    details: dict[str, Any] = {}
    passing = 0
    for family in families:
        candidates = [
            row for row in rows if row["code_family"] == family and abs(float(row["p"]) - required_p) <= 1e-12
        ]
        eligible = []
        for row in candidates:
            passed = bool(
                row.get("exact_fraction", 0.0) >= 1.0
                and row.get("completion_fraction", 0.0) >= float(thresholds["minimum_completion_fraction"])
                and row.get("mean_speedup", 0.0) >= float(thresholds["minimum_mean_total_work_speedup"])
                and row.get("p05_speedup", 0.0) >= float(thresholds["minimum_p05_total_work_speedup"])
                and row.get("median_membership_fraction", 1.0)
                <= float(thresholds["maximum_median_membership_fraction"])
            )
            eligible.append({**row, "pass": passed})
        family_pass = any(row["pass"] for row in eligible)
        passing += int(family_pass)
        details[family] = {"pass": family_pass, "algorithms": eligible}
    return {
        "pass": bool(payload.get("exact_all", False) and passing >= int(thresholds["minimum_passing_families"])),
        "passing_families": passing,
        "family_details": details,
    }


def final_decision(
    exactness: dict[str, Any],
    theory: dict[str, Any],
    boundary: dict[str, Any],
    one_deletion: dict[str, Any],
    insertion: dict[str, Any],
    two_deletion: dict[str, Any],
    code_transfer: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    evaluations = {
        "exactness": {"pass": bool(exactness.get("pass", False))},
        "theory_identity": {"pass": bool(theory.get("pass", False))},
        "boundary": evaluate_boundary(boundary),
        "one_deletion": evaluate_one_deletion(one_deletion),
        "one_insertion": _evaluate_single_algorithm_family(insertion, "one_insertion"),
        "two_deletion": _evaluate_single_algorithm_family(two_deletion, "two_deletion"),
        "code_transfer": evaluate_code_transfer(code_transfer),
    }

    if not evaluations["exactness"]["pass"]:
        verdict = {
            "classification": STOP,
            "stage": "EXACTNESS_OR_CERTIFICATE_FAILURE",
            "reason": "At least one exact likelihood, decoder, insertion/deletion, or certificate audit failed.",
            "authorized_follow_up": "Repair and independently verify the failed correctness audit only. No performance claim is authorized.",
        }
        return verdict, evaluations

    if not evaluations["theory_identity"]["pass"]:
        verdict = {
            "classification": STOP,
            "stage": "THEORY_IMPLEMENTATION_FAILURE",
            "reason": "The finite-rank or posterior-order identities failed their exact audit.",
            "authorized_follow_up": "Repair the mathematical implementation only.",
        }
        return verdict, evaluations

    if not evaluations["boundary"]["pass"] or not evaluations["one_deletion"]["pass"]:
        verdict = {
            "classification": STOP,
            "stage": "CORE_ONE_DELETION_MECHANISM_FAILED",
            "reason": (
                "The proposal's indispensable first problem did not jointly show operational path-multiplicity relevance, "
                "controlled search inflation, and a strong fully accounted exact-work advantage across two code families."
            ),
            "authorized_follow_up": "Archive the broad programme. A narrow exact-decoder or negative/boundary paper may be assessed separately.",
        }
        return verdict, evaluations

    broad_pass = bool(
        evaluations["one_insertion"]["pass"]
        and evaluations["two_deletion"]["pass"]
        and evaluations["code_transfer"]["pass"]
    )
    if not broad_pass:
        verdict = {
            "classification": PIVOT_NARROW,
            "stage": "ONE_DELETION_CORE_ONLY",
            "reason": (
                "The exact one-deletion core is computationally promising, but the evidence does not yet support a reusable "
                "bounded-edit paradigm across insertion, two-deletion, and multiple code-family gates."
            ),
            "surviving_path": (
                "Freeze a narrow one-deletion-plus-substitution claim, complete a primary-source novelty audit, and compare "
                "against strong deletion-specific and priority-first exact baselines before any broader theory."
            ),
            "authorized_follow_up": "One narrowly scoped theorem/algorithm paper only; do not begin the general transducer or hardware programme.",
            "stop_original_claim": True,
        }
        return verdict, evaluations

    verdict = {
        "classification": PIVOT_REVISED,
        "stage": "COMPUTATIONAL_CORE_PASSED_NOVELTY_REFRAME_REQUIRED",
        "reason": (
            "Certified aggregate inverse search shows a broad early computational signal across one deletion, one insertion, "
            "two deletions, and several unmodified code families. However, 'fiber guesswork' is ordinary posterior conditional "
            "guesswork and the elementary rank/certificate results are not a new information-theoretic object."
        ),
        "surviving_path": (
            "Reframe the programme around certified code-modular aggregate inverse search, search-inflation/ambiguity laws, "
            "and a tractability-complexity theorem. Do not claim a new guesswork random variable."
        ),
        "authorized_follow_up": (
            "Proceed only to a new claim contract containing: an independent primary-source novelty audit; strong A*/trellis, "
            "VT/GC+/marker and code-specific baselines; theorem extraction for search inflation or fixed-parameter edit complexity; "
            "and a calibrated synchronization use case."
        ),
        "stop_original_claim": True,
        "warning": "This is evidence of field-defining potential, not proof of novelty, asymptotic superiority, or real-system relevance.",
    }
    return verdict, evaluations


def _fmt(value: Any, digits: int = 4) -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.{digits}g}"
    return str(value)


def write_verdict_files(
    results_dir: Path,
    verdict: dict[str, Any],
    evaluations: dict[str, Any],
    evidence: dict[str, Any],
) -> None:
    payload = {
        **verdict,
        "gate_evaluations": evaluations,
        "evidence_summary": evidence,
        "manual_novelty_gate": "PENDING_INDEPENDENT_PRIMARY_SOURCE_ADJUDICATION",
    }
    write_json(results_dir / "GATE_VERDICT.json", payload)

    lines = [
        "# FIBER-GRAND Kill-Fast Verdict",
        "",
        f"**Classification:** `{verdict['classification']}`",
        "",
        f"**Decision stage:** `{verdict.get('stage', '')}`",
        "",
        f"**Reason:** {verdict.get('reason', '')}",
        "",
        f"**Authorized follow-up:** {verdict.get('authorized_follow_up', '')}",
    ]
    if verdict.get("surviving_path"):
        lines.extend(["", f"**Surviving path:** {verdict['surviving_path']}"])
    if verdict.get("warning"):
        lines.extend(["", f"**Warning:** {verdict['warning']}"])
    lines.extend(
        [
            "",
            "## Gate results",
            "",
            "| Gate | Pass | Key evidence |",
            "|---|---:|---|",
            f"| Exactness/certificates | {_fmt(evaluations['exactness']['pass'])} | {evidence.get('exactness_checks')} audit groups. |",
            f"| Theory identities | {_fmt(evaluations['theory_identity']['pass'])} | G_fib posterior-order identity and random-code rank law. |",
            f"| Path-multiplicity boundary | {_fmt(evaluations['boundary']['pass'])} | {evaluations['boundary'].get('disagreement_configs')} configs with nontrivial pathwise ML disagreement; max median inflation {_fmt(evaluations['boundary'].get('max_median_search_inflation'))}. |",
            f"| One deletion + substitutions | {_fmt(evaluations['one_deletion']['pass'])} | {evaluations['one_deletion'].get('passing_families')} code families pass. |",
            f"| One insertion + substitutions | {_fmt(evaluations['one_insertion']['pass'])} | {evaluations['one_insertion'].get('passing_families')} code families pass. |",
            f"| Two deletions + substitutions | {_fmt(evaluations['two_deletion']['pass'])} | {evaluations['two_deletion'].get('passing_families')} code families pass. |",
            f"| Code-family transfer | {_fmt(evaluations['code_transfer']['pass'])} | {evaluations['code_transfer'].get('passing_families')} families pass at the frozen transfer point. |",
            "| Primary-source novelty | pending | Required before a field-defining claim. Fiber guesswork itself is standard conditional guesswork. |",
            "| Strong specialized baselines | pending | Required next: priority-first/A*, channel-code trellises, VT/GC+/marker/polar synchronization decoders. |",
            "| Calibrated telecom use case | pending | Synthetic edit channels do not establish real-system relevance. |",
            "",
            "## Reference-standard numerical signal",
            "",
            f"- Maximum observed pathwise-first-hit disagreement fraction: {_fmt(evidence.get('maximum_pathwise_disagreement_fraction'))}.",
            f"- Maximum median distinct-candidate search inflation over ideal posterior rank: {_fmt(evidence.get('maximum_median_search_inflation'))}.",
            f"- One deletion at p=0.05 (best frozen result): operation-count speedup {_fmt((evidence.get('best_one_deletion_p005') or {}).get('mean_speedup'))}, median wall-clock speedup {_fmt((evidence.get('best_one_deletion_p005') or {}).get('median_wall_speedup'))}.",
            f"- One insertion at p=0.05: operation-count speedup {_fmt((evidence.get('best_one_insertion_p005') or {}).get('mean_speedup'))}, median wall-clock speedup {_fmt((evidence.get('best_one_insertion_p005') or {}).get('median_wall_speedup'))}.",
            f"- Two deletions at p=0.05: operation-count speedup {_fmt((evidence.get('best_two_deletion_p005') or {}).get('mean_speedup'))}, median Python wall-clock speedup {_fmt((evidence.get('best_two_deletion_p005') or {}).get('median_wall_speedup'))}.",
            "",
            (
                "The two-deletion probe converts abstract-work reduction into only a modest largest-n Python gain, based on a "
                "deliberately small kill-fast sample; it is not evidence of scalable superiority. "
                if float((evidence.get("best_two_deletion_p005") or {}).get("median_wall_speedup", 0.0)) > 1.0
                else "The two-deletion Python prototype reduces the frozen abstract work count but is not yet faster in wall-clock time. "
            )
            + "This is why the result authorizes theorem/baseline/optimization work only, not hardware or a general-transducer claim.",
            "",
            "## Interpretation boundary",
            "",
            "The gate can refute correctness and reveal finite-block computational mechanisms. It cannot prove novelty, "
            "a universal asymptotic work separation, or real-world value. A revised-track verdict stops the original wording "
            "and authorizes only the explicitly frozen next claim contract.",
            "",
        ]
    )
    (results_dir / "GATE_REPORT.md").write_text("\n".join(lines), encoding="utf-8")

    for name in ("STOP_COMMAND.txt", "PIVOT_COMMAND.txt", "CONTINUE_COMMAND.txt"):
        path = results_dir / name
        if path.exists():
            path.unlink()
    classification = verdict["classification"]
    if classification == STOP:
        (results_dir / "STOP_COMMAND.txt").write_text(
            "STOP_FIELD_DEFINING_FIBER_PROGRAMME\n"
            f"STAGE={verdict.get('stage','')}\n"
            f"REASON={verdict.get('reason','')}\n"
            "DO_NOT_START=deep_profile,general_transducer_theory,hardware,real_world_trials,multi_year_programme\n",
            encoding="utf-8",
        )
    elif classification in (PIVOT_REVISED, PIVOT_NARROW):
        (results_dir / "PIVOT_COMMAND.txt").write_text(
            "STOP_ORIGINAL_FIBER_GRAND_CLAIM_SET\n"
            f"PIVOT_CLASS={classification}\n"
            f"PIVOT_STAGE={verdict.get('stage','')}\n"
            f"SURVIVING_PATH={verdict.get('surviving_path','')}\n"
            "REQUIRE_NEW_CLAIM_CONTRACT=YES\n",
            encoding="utf-8",
        )
    else:
        (results_dir / "CONTINUE_COMMAND.txt").write_text(
            "CONTINUE_ONLY_TO_INDEPENDENT_NOVELTY_AND_THEOREM_EXTRACTION\n"
            "FIELD_DEFINING_STATUS=NOT_YET_ESTABLISHED\n",
            encoding="utf-8",
        )

    handoff = [
        "# Analysis Handoff",
        "",
        "Read in this order:",
        "",
        "1. `GATE_VERDICT.json`",
        "2. `GATE_REPORT.md`",
        "3. `01_exactness_audit.json`",
        "4. `02_theory_boundary.json`",
        "5. `03_boundary_audit.json` and summary CSV",
        "6. `04_one_deletion_scaling.json` and summary CSV",
        "7. `05_one_insertion_scaling.json` and summary CSV",
        "8. `06_two_deletion_scaling.json` and summary CSV",
        "9. `07_code_transfer.json` and summary CSV",
        "10. `RESULTS_SHA256.txt`",
        "",
        "Exactly one command file is authoritative for research-investment discipline.",
        "",
    ]
    (results_dir / "ANALYSIS_HANDOFF.md").write_text("\n".join(handoff), encoding="utf-8")
