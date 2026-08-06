from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .utils import read_json, write_json

CONTINUE = "CONTINUE_HIGH_POTENTIAL_FIXED_EDIT_FIBER_TRACK_REAL_TRACE_REQUIRED"
NARROW = "NARROW_TO_FIXED_EDIT_THEOREM_AND_ALGORITHM_PAPER"
STOP = "STOP_FIBER_FIELD_DEFINING_PROGRAMME"


def evaluate(results_dir: Path) -> dict[str, Any]:
    contract = read_json(results_dir / "00_decision_contract.json")
    exact = read_json(results_dir / "00_exactness_audit.json")
    novelty = read_json(results_dir / "01_novelty_adjudication.json")
    moment = read_json(results_dir / "02_moment_tail_gate.json")
    phase = read_json(results_dir / "03_phase_diagram_gate.json")
    benchmark = read_json(results_dir / "04_primary_benchmark_gate.json")
    vt = read_json(results_dir / "05_vt_boundary_gate.json")
    physical = read_json(results_dir / "07_physical_plausibility_gate.json")

    novelty_pass = novelty.get("overall_classification") == contract["novelty"]["required_adjudication"]
    exact_pass = bool(exact.get("pass", False) and benchmark.get("exact_agreement_all", False) and benchmark.get("completion_all", False))

    mt = contract["moment_tail_theory"]
    moment_pass = bool(
        moment["maximum_largest_n_reveal_moment_gap"] <= mt["maximum_largest_n_reveal_moment_gap"]
        and moment["maximum_largest_n_certificate_upper_gap"] <= mt["maximum_largest_n_certificate_upper_gap"]
        and moment["maximum_largest_n_quantile_gap"] <= mt["maximum_largest_n_quantile_gap"]
        and moment["gap_decrease_fraction"] >= mt["require_gap_decrease_fraction"]
        and moment["variational_identity_max_error"] <= 1e-10
    )
    phase_pass = bool(phase.get("mean_threshold_below_typical_all", False))

    summary = pd.read_csv(results_dir / "04_primary_summary.csv")
    slopes = pd.read_csv(results_dir / "04_primary_slopes.csv")
    largest_n = int(summary["n"].max())
    favorable = summary[(summary["n"] == largest_n) & summary["predicted_typical_favorable"]]
    bc = contract["benchmark"]
    median_wins = int((favorable["median_fiber_over_best_wall"] <= 0.8).sum())
    p95_wins = int((favorable["p95_fiber_over_best_wall"] <= bc["maximum_p95_ratio_for_at_least_two_configs"]).sum())
    favorable_slopes = slopes[slopes["predicted_typical_favorable"]]
    nondegrading_slopes = int(
        (favorable_slopes["fiber_minus_best_slope"] <= bc["maximum_favorable_slope_disadvantage"]).sum()
    )
    benchmark_pass = bool(
        benchmark.get("codebook_free", False)
        and len(favorable) >= bc["minimum_predicted_favorable_configs"]
        and median_wins >= bc["minimum_configs_median_wall_win_1p25"]
        and p95_wins >= 2
        and float(favorable["median_fiber_over_best_wall"].max())
        <= bc["maximum_median_ratio_in_any_predicted_favorable_config"]
        and nondegrading_slopes >= bc["minimum_favorable_configs_with_nondegrading_slope"]
    )

    vt_pass = bool(vt.get("agreement_all", False) and vt.get("linear_candidate_check_identity", False))
    pc = contract["physical_plausibility"]
    physical_pass = bool(
        physical["exactly_one_slip_fraction"] >= pc["minimum_exactly_one_slip_fraction"]
        and physical["heldout_deletion_position_tv_to_fitted"] <= pc["maximum_deletion_position_tv_to_fitted"]
        and physical["absolute_p_fit_error"] <= pc["maximum_absolute_p_fit_error"]
        and abs(physical["heldout_error_lag1_correlation"]) <= pc["maximum_error_lag1_correlation"]
    )
    measured_present = physical["measured_trace_audit"].get("status") == "MEASURED_TRACE_PRESENT"

    evidence = {
        "novelty_pass": novelty_pass,
        "exactness_pass": exact_pass,
        "moment_tail_theory_pass": moment_pass,
        "phase_diagram_pass": phase_pass,
        "benchmark_pass": benchmark_pass,
        "vt_boundary_pass": vt_pass,
        "synthetic_physical_plausibility_pass": physical_pass,
        "measured_real_trace_present": measured_present,
        "largest_benchmark_n": largest_n,
        "predicted_favorable_largest_n_configs": int(len(favorable)),
        "median_wall_wins_1p25": median_wins,
        "p95_ratio_pass_configs": p95_wins,
        "nondegrading_favorable_slope_configs": nondegrading_slopes,
        "largest_n_favorable_rows": favorable.to_dict(orient="records"),
        "moment_summary": moment,
        "physical_summary": physical,
    }

    if not exact_pass or not novelty_pass or not moment_pass or not phase_pass:
        verdict = {
            "classification": STOP,
            "stage": "FOUNDATIONAL_NOVELTY_CORRECTNESS_OR_THEORY_FAILURE",
            "reason": "At least one indispensable novelty, exactness, moment/tail theorem, or phase-diagram requirement failed.",
            "authorized_follow_up": "Repair only a demonstrable implementation defect. Otherwise stop the field-defining programme immediately.",
        }
    elif not benchmark_pass:
        verdict = {
            "classification": NARROW,
            "stage": "THEORY_SURVIVES_BUT_SCALABLE_DECODER_ADVANTAGE_NOT_ESTABLISHED",
            "reason": "The narrowed fixed-edit theory remains publishable, but FIBER did not show the preregistered competitive scaling and tail signal against strong exact code-specific searches.",
            "authorized_follow_up": "Prepare a fixed-edit theorem and exact-algorithm paper; stop the multi-year field-defining investment.",
        }
    elif not vt_pass or not physical_pass:
        verdict = {
            "classification": NARROW,
            "stage": "SPECIALIZATION_OR_PHYSICAL_PLAUSIBILITY_FAILURE",
            "reason": "The main benchmark passed, but an honest specialized-code boundary or the calibrated timing-slip plausibility gate failed.",
            "authorized_follow_up": "Narrow to the exact synthetic fixed-edit result and stop broad field-defining claims.",
        }
    else:
        verdict = {
            "classification": CONTINUE,
            "stage": "NARROW_NOVELTY_THEORY_AND_COMPUTATIONAL_POTENTIAL_CONFIRMED",
            "reason": (
                "The narrowed fixed-edit synthesis survived the primary-source audit, exact certificate checks, new Renyi moment/tail theory, "
                "phase-diagram audit, and codebook-free comparison with two exact code-specific searches in the predicted favorable region."
            ),
            "authorized_follow_up": (
                "Proceed to a paper-grade implementation and one measured post-front-end trace campaign. Field-defining status remains unestablished until "
                "the measured real-system gate and independent external proof/novelty reviews pass."
            ),
        }

    payload = {
        **verdict,
        "field_defining_status": "HIGH_POTENTIAL_NOT_ESTABLISHED" if verdict["classification"] == CONTINUE else "NOT_SUPPORTED",
        "stop_if_next_gate_fails": True,
        "evidence_summary": evidence,
    }
    write_json(results_dir / "GATE_VERDICT.json", payload)

    for name in ("STOP_COMMAND.txt", "NARROW_COMMAND.txt", "CONTINUE_COMMAND.txt"):
        path = results_dir / name
        if path.exists():
            path.unlink()
    if payload["classification"] == CONTINUE:
        (results_dir / "CONTINUE_COMMAND.txt").write_text(
            "CONTINUE_HIGH_POTENTIAL_FIXED_EDIT_FIBER_TRACK_REAL_TRACE_REQUIRED\n"
            "FIELD_DEFINING_STATUS=HIGH_POTENTIAL_NOT_ESTABLISHED\n"
            "NEXT_REQUIRED=independent_external_proof_review,measured_post_front_end_trace,paper_grade_compiled_benchmarks\n"
            "STOP_IF_NEXT_GATE_FAILS=YES\n",
            encoding="utf-8",
        )
    elif payload["classification"] == NARROW:
        (results_dir / "NARROW_COMMAND.txt").write_text(
            "NARROW_TO_FIXED_EDIT_THEOREM_AND_ALGORITHM_PAPER\n"
            "MULTI_YEAR_FIELD_DEFINING_PROGRAMME=STOP\n",
            encoding="utf-8",
        )
    else:
        (results_dir / "STOP_COMMAND.txt").write_text(
            "STOP_FIBER_FIELD_DEFINING_PROGRAMME\n"
            "DO_NOT_RUN_DEEP_PROFILE\n"
            "DO_NOT_BEGIN_HARDWARE_OR_GENERAL_TRANSDUCER_WORK\n",
            encoding="utf-8",
        )
    return payload


def write_report(results_dir: Path, verdict: dict[str, Any]) -> None:
    e = verdict["evidence_summary"]
    lines = [
        "# FIBER-GRAND Final Potential Gate",
        "",
        f"**Classification:** `{verdict['classification']}`",
        "",
        f"**Stage:** `{verdict['stage']}`",
        "",
        f"**Field-defining status:** `{verdict['field_defining_status']}`",
        "",
        f"**Reason:** {verdict['reason']}",
        "",
        f"**Authorized follow-up:** {verdict['authorized_follow_up']}",
        "",
        "## Gate summary",
        "",
        f"- Narrow novelty: `{e['novelty_pass']}`",
        f"- Exactness and certificates: `{e['exactness_pass']}`",
        f"- Moment/tail theory: `{e['moment_tail_theory_pass']}`",
        f"- Typical/mean phase diagram: `{e['phase_diagram_pass']}`",
        f"- Strong exact benchmark: `{e['benchmark_pass']}`",
        f"- True linear-time VT boundary: `{e['vt_boundary_pass']}`",
        f"- Synthetic physical plausibility: `{e['synthetic_physical_plausibility_pass']}`",
        f"- Measured real trace present: `{e['measured_real_trace_present']}`",
        "",
        "## Interpretation boundary",
        "",
        "A positive result confirms high potential only for the fixed-edit, low-substitution, code-modular track. It does not restore the withdrawn fiber-guesswork or unrestricted-transducer claims and does not establish retrospective field impact.",
        "",
    ]
    (results_dir / "GATE_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
