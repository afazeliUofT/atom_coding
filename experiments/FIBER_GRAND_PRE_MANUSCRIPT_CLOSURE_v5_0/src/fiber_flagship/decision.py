from __future__ import annotations

from pathlib import Path
from typing import Any

from .utils import read_json, write_json

PROCEED = "AUTHORIZE_FULL_TIT_MANUSCRIPT_AND_EXTERNAL_REVIEW"
NARROW = "NARROW_TO_FIXED_EDIT_TIT_THEORY_ALGORITHM_PAPER"
STOP = "STOP_FLAGSHIP_PERFORMANCE_CLAIM"


def evaluate(results_dir: Path) -> dict[str, Any]:
    exact = read_json(results_dir / "00_exactness_audit.json")
    candidate = read_json(results_dir / "01_candidate_theory_gate.json")
    moment = read_json(results_dir / "02_moment_tail_gate.json")
    phase = read_json(results_dir / "03_phase_diagram_gate.json")
    compiled = read_json(results_dir / "03_compiled_gate.json")
    vt = read_json(results_dir / "05_vt_boundary_gate.json")
    trace = read_json(results_dir / "04_measured_trace_gate.json")
    contract = read_json(results_dir / "00_decision_contract.json")

    exact_pass = bool(exact.get("pass", False))
    candidate_pass = bool(
        candidate.get("candidate_volume_bounds_pass", False)
        and candidate.get("exact_ambiguity_degree_pass", False)
        and candidate.get("oracle_to_fiber_inflation_bounds_pass", False)
        and candidate.get("known_cardinality_refinement_pass", False)
    )
    mt = contract["moment_tail_theory"]
    moment_pass = bool(
        moment["maximum_largest_n_reveal_moment_gap"]
        <= mt["maximum_largest_n_reveal_moment_gap"]
        and moment["maximum_largest_n_certificate_upper_gap"]
        <= mt["maximum_largest_n_certificate_upper_gap"]
        and moment["maximum_largest_n_quantile_gap"]
        <= mt["maximum_largest_n_quantile_gap"]
        and moment["gap_decrease_fraction"] >= mt["require_gap_decrease_fraction"]
        and moment["variational_identity_max_error"] <= 1e-10
    )
    phase_pass = bool(phase.get("mean_threshold_below_typical_all", False))
    compiled_pass = bool(compiled.get("pass", False))
    vt_pass = bool(
        vt.get("agreement_all", False) and vt.get("linear_candidate_check_identity", False)
    )
    measured_present = trace.get("status") == "MEASURED_TRACE_PRESENT"

    evidence = {
        "exactness_pass": exact_pass,
        "candidate_volume_oracle_and_known_cardinality_theory_pass": candidate_pass,
        "moment_tail_theory_pass": moment_pass,
        "phase_diagram_pass": phase_pass,
        "compiled_baseline_and_tail_closure_pass": compiled_pass,
        "trellis_coverage_pass": bool(compiled.get("trellis_coverage_pass", False)),
        "positive_error_evidence_pass": bool(
            compiled.get("positive_error_evidence_pass", False)
        ),
        "minimum_key_trials_pass": bool(compiled.get("minimum_key_trials_pass", False)),
        "vt_specialization_boundary_pass": vt_pass,
        "measured_real_trace_present": measured_present,
        "candidate_theory": candidate,
        "compiled_summary": compiled,
        "trace_summary": trace,
    }

    if not (exact_pass and candidate_pass and moment_pass and phase_pass):
        verdict = {
            "classification": STOP,
            "stage": "FOUNDATIONAL_CORRECTNESS_OR_THEORY_FAILURE",
            "reason": (
                "At least one indispensable exactness, candidate-volume/oracle, known-cardinality "
                "refinement, moment/tail, or phase-diagram requirement failed."
            ),
            "authorized_follow_up": (
                "Repair only a demonstrable implementation defect. Otherwise stop the flagship-performance track."
            ),
        }
    elif not compiled_pass:
        verdict = {
            "classification": NARROW,
            "stage": "THEORY_SURVIVES_BUT_STRONG_BASELINE_OR_TAIL_CLOSURE_FAILED",
            "reason": (
                "The fixed-edit theory and exact decoder remain publishable, but the advantage did not "
                "survive the enabled feasible syndrome-trellis references, supported p99 statistics, or "
                "positive-substitution conditioning."
            ),
            "authorized_follow_up": (
                "Prepare a rigorous IEEE TIT theory/algorithm paper without a flagship performance claim; "
                "retain the negative or boundary results and stop further performance escalation."
            ),
        }
    elif not vt_pass:
        verdict = {
            "classification": NARROW,
            "stage": "SPECIALIZATION_BOUNDARY_FAILURE",
            "reason": (
                "The main closure passed, but the exact linear-time VT specialization boundary was not reproduced."
            ),
            "authorized_follow_up": (
                "Narrow to the linear-code deletion/BSC theorem and decoder result."
            ),
        }
    else:
        verdict = {
            "classification": PROCEED,
            "stage": "PRE_MANUSCRIPT_STRONG_BASELINE_AND_TAIL_CLOSURE_PASSED",
            "reason": (
                "The narrowed fixed-edit programme remains exact, the oracle-relative theory survives a "
                "known-cardinality refinement, the feasible syndrome-trellis baseline is enabled at the "
                "required larger-length points, and the compiled advantage persists with supported p99 "
                "and positive-error-conditioned evidence."
            ),
            "authorized_follow_up": (
                "Freeze the full IEEE Transactions on Information Theory manuscript and a minimal standalone "
                "external-review package. Independent theorem-by-theorem proof and novelty review remain "
                "mandatory before submission."
            ),
        }

    payload = {
        **verdict,
        "quality_flagship_paper_potential": verdict["classification"] == PROCEED,
        "field_defining_status": "NOT_ESTABLISHED",
        "real_system_claim_status": (
            "SUPPORTED_ONLY_IF_MEASURED_TRACE_PRESENT"
            if measured_present
            else "PENDING_MEASURED_TRACE"
        ),
        "stop_if_external_proof_or_novelty_review_fails": True,
        "evidence_summary": evidence,
    }
    write_json(results_dir / "GATE_VERDICT.json", payload)

    for name in ("STOP_COMMAND.txt", "NARROW_COMMAND.txt", "CONTINUE_COMMAND.txt"):
        path = results_dir / name
        if path.exists():
            path.unlink()
    if payload["classification"] == PROCEED:
        (results_dir / "CONTINUE_COMMAND.txt").write_text(
            "AUTHORIZE_FULL_TIT_MANUSCRIPT_AND_EXTERNAL_REVIEW\n"
            "QUALITY_FLAGSHIP_PAPER_POTENTIAL=YES\n"
            "FIELD_DEFINING_STATUS=NOT_ESTABLISHED\n"
            "NEXT_REQUIRED=full_TIT_manuscript,minimal_external_review_package,independent_proof_review,independent_novelty_review\n"
            "REAL_SYSTEM_CLAIMS_REQUIRE_MEASURED_TRACE=YES\n"
            "STOP_IF_EXTERNAL_REVIEW_FINDS_BLOCKING_DEFECT=YES\n",
            encoding="utf-8",
        )
    elif payload["classification"] == NARROW:
        (results_dir / "NARROW_COMMAND.txt").write_text(
            "NARROW_TO_FIXED_EDIT_TIT_THEORY_ALGORITHM_PAPER\n"
            "FLAGSHIP_PERFORMANCE_CLAIM=STOP\n"
            "FIELD_DEFINING_STATUS=NOT_SUPPORTED\n",
            encoding="utf-8",
        )
    else:
        (results_dir / "STOP_COMMAND.txt").write_text(
            "STOP_FLAGSHIP_PERFORMANCE_CLAIM\n"
            "DO_NOT_BEGIN_FULL_FLAGSHIP_MANUSCRIPT_OR_HARDWARE_WORK\n",
            encoding="utf-8",
        )
    return payload


def write_report(results_dir: Path, verdict: dict[str, Any]) -> None:
    e = verdict["evidence_summary"]
    lines = [
        "# FIBER-GRAND Pre-Manuscript Closure Gate",
        "",
        f"**Classification:** `{verdict['classification']}`",
        "",
        f"**Stage:** `{verdict['stage']}`",
        "",
        f"**Reason:** {verdict['reason']}",
        "",
        f"**Authorized follow-up:** {verdict['authorized_follow_up']}",
        "",
        "## Gate summary",
        "",
        f"- Exactness: `{e['exactness_pass']}`",
        f"- Candidate-volume/oracle/cardinality theory: `{e['candidate_volume_oracle_and_known_cardinality_theory_pass']}`",
        f"- Moment/tail theory: `{e['moment_tail_theory_pass']}`",
        f"- Typical/mean phase diagram: `{e['phase_diagram_pass']}`",
        f"- Feasible trellis coverage: `{e['trellis_coverage_pass']}`",
        f"- Positive-error-conditioned evidence: `{e['positive_error_evidence_pass']}`",
        f"- Supported key sample sizes: `{e['minimum_key_trials_pass']}`",
        f"- Compiled closure benchmark: `{e['compiled_baseline_and_tail_closure_pass']}`",
        f"- Linear-time VT boundary: `{e['vt_specialization_boundary_pass']}`",
        f"- Measured post-front-end trace: `{e['measured_real_trace_present']}`",
        "",
        "## Claim boundary",
        "",
        "A positive result authorizes a full theory/algorithm manuscript and external review only. "
        "It does not restore generic fiber-guesswork, unrestricted-transducer, FPT, processor-cycle-optimality, "
        "real-system, or field-defining claims.",
        "",
    ]
    (results_dir / "GATE_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
