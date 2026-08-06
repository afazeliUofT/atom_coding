from __future__ import annotations

from pathlib import Path
from typing import Any

from .utils import read_json, write_json

PROCEED = "PROCEED_TO_FLAGSHIP_TIT_MANUSCRIPT_AND_EXTERNAL_REVIEW"
NARROW = "NARROW_TO_STRONG_FIXED_EDIT_THEORY_AND_ALGORITHM_PAPER"
STOP = "STOP_FIBER_FLAGSHIP_TRACK"


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
    )
    mt = contract["moment_tail_theory"]
    moment_pass = bool(
        moment["maximum_largest_n_reveal_moment_gap"] <= mt["maximum_largest_n_reveal_moment_gap"]
        and moment["maximum_largest_n_certificate_upper_gap"] <= mt["maximum_largest_n_certificate_upper_gap"]
        and moment["maximum_largest_n_quantile_gap"] <= mt["maximum_largest_n_quantile_gap"]
        and moment["gap_decrease_fraction"] >= mt["require_gap_decrease_fraction"]
        and moment["variational_identity_max_error"] <= 1e-10
    )
    phase_pass = bool(phase.get("mean_threshold_below_typical_all", False))
    compiled_pass = bool(compiled.get("pass", False))
    vt_pass = bool(vt.get("agreement_all", False) and vt.get("linear_candidate_check_identity", False))
    measured_present = trace.get("status") == "MEASURED_TRACE_PRESENT"

    evidence = {
        "exactness_pass": exact_pass,
        "candidate_volume_and_oracle_theory_pass": candidate_pass,
        "moment_tail_theory_pass": moment_pass,
        "phase_diagram_pass": phase_pass,
        "compiled_benchmark_pass": compiled_pass,
        "vt_specialization_boundary_pass": vt_pass,
        "measured_real_trace_present": measured_present,
        "candidate_theory": candidate,
        "compiled_summary": compiled,
        "trace_summary": trace,
    }

    if not (exact_pass and candidate_pass and moment_pass and phase_pass):
        verdict = {
            "classification": STOP,
            "stage": "FOUNDATIONAL_THEORY_OR_CORRECTNESS_FAILURE",
            "reason": (
                "At least one indispensable exactness, candidate-volume/oracle-competitiveness, moment/tail, "
                "or phase-diagram requirement failed."
            ),
            "authorized_follow_up": "Repair only a demonstrable implementation defect. Otherwise stop the flagship track.",
        }
    elif not compiled_pass:
        verdict = {
            "classification": NARROW,
            "stage": "THEORY_SURVIVES_COMPILED_FLAGSHIP_SIGNAL_NOT_ESTABLISHED",
            "reason": (
                "The fixed-edit theorems and exact decoder remain publishable, but the equally compiled comparison did not "
                "provide the preregistered robust finite-length signal required for a flagship manuscript."
            ),
            "authorized_follow_up": (
                "Prepare a strong fixed-edit theory/algorithm paper with honest negative performance boundaries; "
                "stop the flagship-performance claim."
            ),
        }
    elif not vt_pass:
        verdict = {
            "classification": NARROW,
            "stage": "SPECIALIZATION_BOUNDARY_FAILURE",
            "reason": "The main gate passed, but the exact VT specialization boundary was not reproduced.",
            "authorized_follow_up": "Narrow to the linear-code one-deletion/BSC theorem and decoder result.",
        }
    else:
        verdict = {
            "classification": PROCEED,
            "stage": "FLAGSHIP_THEORY_AND_COMPILED_MECHANISM_GATE_PASSED",
            "reason": (
                "The narrowed fixed-edit programme now has an exact code-modular decoder, a strict certificate, "
                "candidate-volume and membership-oracle competitiveness bounds, moment/tail laws, a rate-complexity phase diagram, "
                "and a reproducible compiled advantage in the predicted low-substitution region."
            ),
            "authorized_follow_up": (
                "Proceed immediately to a full IEEE Transactions on Information Theory manuscript and independent external proof/novelty review. "
                "A measured trace is optional for the theory paper but mandatory before any real-system or field-defining-impact claim."
            ),
        }

    payload = {
        **verdict,
        "quality_flagship_paper_potential": verdict["classification"] == PROCEED,
        "field_defining_status": "NOT_ESTABLISHED",
        "real_system_claim_status": "SUPPORTED_ONLY_IF_MEASURED_TRACE_PRESENT" if measured_present else "PENDING_MEASURED_TRACE",
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
            "PROCEED_TO_FLAGSHIP_TIT_MANUSCRIPT_AND_EXTERNAL_REVIEW\n"
            "QUALITY_FLAGSHIP_PAPER_POTENTIAL=YES\n"
            "FIELD_DEFINING_STATUS=NOT_ESTABLISHED\n"
            "NEXT_REQUIRED=full_TIT_manuscript,independent_proof_review,independent_novelty_review\n"
            "REAL_SYSTEM_CLAIMS_REQUIRE_MEASURED_TRACE=YES\n"
            "STOP_IF_EXTERNAL_REVIEW_FINDS_BLOCKING_DEFECT=YES\n",
            encoding="utf-8",
        )
    elif payload["classification"] == NARROW:
        (results_dir / "NARROW_COMMAND.txt").write_text(
            "NARROW_TO_STRONG_FIXED_EDIT_THEORY_AND_ALGORITHM_PAPER\n"
            "FLAGSHIP_PERFORMANCE_CLAIM=STOP\n"
            "FIELD_DEFINING_STATUS=NOT_SUPPORTED\n",
            encoding="utf-8",
        )
    else:
        (results_dir / "STOP_COMMAND.txt").write_text(
            "STOP_FIBER_FLAGSHIP_TRACK\n"
            "DO_NOT_BEGIN_FULL_MANUSCRIPT_OR_HARDWARE_WORK\n",
            encoding="utf-8",
        )
    return payload


def write_report(results_dir: Path, verdict: dict[str, Any]) -> None:
    e = verdict["evidence_summary"]
    lines = [
        "# FIBER-GRAND Flagship-Paper Readiness Gate",
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
        f"- Candidate-volume and L0-oracle theorem audit: `{e['candidate_volume_and_oracle_theory_pass']}`",
        f"- Moment/tail theory: `{e['moment_tail_theory_pass']}`",
        f"- Typical/mean phase diagram: `{e['phase_diagram_pass']}`",
        f"- Same-toolchain compiled exact benchmark: `{e['compiled_benchmark_pass']}`",
        f"- Linear-time VT specialization boundary: `{e['vt_specialization_boundary_pass']}`",
        f"- Measured post-front-end trace: `{e['measured_real_trace_present']}`",
        "",
        "## Claim boundary",
        "",
        "A positive result supports a quality theory/algorithm flagship-paper programme for fixed edit count and low substitution probability. "
        "It does not restore the withdrawn generic fiber-guesswork, unrestricted-transducer, FPT, or real-system claims.",
        "",
    ]
    (results_dir / "GATE_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
