from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .utils import read_json, write_json

CONTINUE = "CONTINUE_REVISED_FIBER_TRACK_TO_FINAL_GATE"
NARROW = "NARROW_TO_FIXED_EDIT_FOUNDATIONS_STOP_FIELD_DEFINING"
STOP = "STOP_FIBER_FIELD_DEFINING_PROGRAMME"


def evaluate(results_dir: Path) -> dict[str, Any]:
    contract = read_json(results_dir / "00_decision_contract.json")
    exactness = read_json(results_dir / "01_exactness_audit.json")
    primary = read_json(results_dir / "02_primary_gate.json")
    two = read_json(results_dir / "03_two_deletion_gate.json")
    vt = read_json(results_dir / "04_vt_gate.json")
    theorem = read_json(results_dir / "05_shell_theorem_gate.json")
    summary = pd.read_csv(results_dir / "02_primary_summary.csv")

    exact_pass = bool(exactness.get("pass", False) and primary.get("exact_agreement_all", False))
    codebook_free = bool(primary.get("codebook_free", False))
    theorem_summary = pd.read_csv(results_dir / "05_shell_theorem_summary.csv")
    max_excess_by_n = theorem_summary.groupby("n")["p95_excess_over_h2"].max().sort_index()
    decreasing_excess = bool(all(
        later <= earlier + 1e-12
        for earlier, later in zip(max_excess_by_n.to_numpy()[:-1], max_excess_by_n.to_numpy()[1:])
    ))
    theorem_pass = bool(
        theorem.get("all_inequalities", False)
        and two.get("all_observed_below_theorem_bound", False)
        and theorem.get("maximum_p95_excess_over_h2_at_largest_n", 99.0)
        <= float(contract["shell_theorem"]["maximum_large_n_normalized_bound_minus_h2"])
        and decreasing_excess
    )

    sb = contract["strong_baseline"]
    largest_n = int(summary["n"].max())
    required_rows = []
    for family in sb["required_families"]:
        for p in sb["required_probabilities"]:
            candidates = summary[
                (summary["family"] == family)
                & (abs(summary["p"] - float(p)) <= 1e-12)
                & (summary["n"] == largest_n)
            ]
            if not candidates.empty:
                required_rows.append(candidates.iloc[0].to_dict())

    baseline_complete = len(required_rows) == len(sb["required_families"]) * len(sb["required_probabilities"])
    all_completion = baseline_complete and all(
        row["agreement_fraction"] >= 1.0
        and row["fiber_completion_fraction"] >= float(sb["minimum_completion_fraction"])
        and row["trellis_completion_fraction"] >= float(sb["minimum_completion_fraction"])
        and row["prefix_completion_fraction"] >= float(sb["minimum_completion_fraction"])
        and row["trials"] >= int(sb["minimum_largest_n_trials_per_config"])
        for row in required_rows
    )
    ratio_ok = baseline_complete and all(
        row["median_fiber_over_best_wall"] <= float(sb["maximum_median_wall_ratio_fiber_over_best_exact_baseline"])
        for row in required_rows
    )
    fiber_wins = sum(
        row["median_best_over_fiber_wall"] >= 1.25
        for row in required_rows
    )
    trellis_dominates_5x = sum(
        row["median_fiber_over_best_wall"] >= 5.0
        for row in required_rows
    )
    strong_baseline_pass = bool(
        all_completion
        and ratio_ok
        and fiber_wins >= int(sb["minimum_configs_with_fiber_wall_win_1p25"])
        and trellis_dominates_5x <= int(sb["maximum_configs_with_trellis_wall_win_5x"])
    )

    vt_contract = contract["vt_specialized_baseline"]
    vt_rows = vt.get("summary", [])
    vt_pass = bool(
        vt.get("agreement_all", False)
        and vt_rows
        and max(float(row["median_fiber_over_vt_wall"]) for row in vt_rows)
        <= float(vt_contract["maximum_median_wall_ratio_fiber_over_vt"])
    )

    two_contract = contract["two_deletion_diagnostic"]
    two_rows = two.get("summary", [])
    two_pass = bool(
        two_rows
        and min(float(row["completion_fraction"]) for row in two_rows)
        >= float(two_contract["minimum_completion_fraction"])
        and max(float(row["median_membership_fraction"]) for row in two_rows)
        <= float(two_contract["maximum_median_membership_fraction"])
    )

    evidence = {
        "exactness_pass": exact_pass,
        "codebook_free_pass": codebook_free,
        "shell_theorem_pass": theorem_pass,
        "shell_theorem_max_p95_excess_by_n": {str(int(k)): float(v) for k, v in max_excess_by_n.items()},
        "shell_theorem_excess_decreasing": decreasing_excess,
        "strong_baseline_pass": strong_baseline_pass,
        "strong_baseline_largest_n": largest_n,
        "required_primary_configs_found": len(required_rows),
        "fiber_wall_wins_1p25": int(fiber_wins),
        "trellis_dominates_5x_configs": int(trellis_dominates_5x),
        "largest_n_primary_rows": required_rows,
        "vt_specialized_pass": vt_pass,
        "two_deletion_diagnostic_pass": two_pass,
        "manual_novelty_gate": "PENDING_INDEPENDENT_PRIMARY_SOURCE_ADJUDICATION",
        "real_impairment_gate": "PENDING_CALIBRATED_POST_FRONT_END_TRACE",
    }

    if not exact_pass or not theorem_pass or not codebook_free:
        verdict = {
            "classification": STOP,
            "stage": "CORRECTNESS_THEOREM_OR_CODEBOOK_FREE_FAILURE",
            "reason": "At least one indispensable exactness, deterministic shell-certificate, or codebook-free execution requirement failed.",
            "authorized_follow_up": "Repair only a demonstrable implementation defect. Otherwise stop immediately.",
        }
    elif trellis_dominates_5x == len(required_rows) and required_rows:
        verdict = {
            "classification": STOP,
            "stage": "STRONG_CODE_SPECIFIC_BASELINE_DOMINATES",
            "reason": "The exact syndrome-trellis aggregate baseline dominates the membership-only decoder by at least 5x in every primary largest-block configuration.",
            "authorized_follow_up": "Archive the field-defining programme. A narrow boundary or algorithm paper may be evaluated separately.",
        }
    elif strong_baseline_pass and vt_pass and two_pass:
        verdict = {
            "classification": CONTINUE,
            "stage": "DECODER_THEOREM_AND_STRONG_BASELINE_GATE_PASSED",
            "reason": (
                "The revised membership-only decoder remained exact and competitive against two independent code-specific exact searches (syndrome-trellis aggregation and prefix A*), "
                "the fixed-edit shell theorem passed its deterministic audits, and transfer to a specialized VT baseline and two deletions remained controlled."
            ),
            "authorized_follow_up": (
                "Proceed only to the final field-defining gate: independent primary-source novelty adjudication, independent proof review, "
                "additional specialized synchronization-system baselines, and one calibrated post-front-end synchronization impairment."
            ),
        }
    else:
        verdict = {
            "classification": NARROW,
            "stage": "FOUNDATIONAL_RESULT_SURVIVES_BUT_FIELD_DEFINING_MECHANISM_NOT_ESTABLISHED",
            "reason": (
                "The exact fixed-edit theory or algorithm remains publishable, but the membership-only decoder did not jointly pass the strong-baseline, VT, and two-deletion requirements needed for a field-defining continuation."
            ),
            "authorized_follow_up": "Prepare a bounded-edit theorem/boundary paper and stop the multi-year field-defining programme.",
        }

    payload = {**verdict, "evidence_summary": evidence}
    write_json(results_dir / "GATE_VERDICT.json", payload)

    command_paths = [results_dir / name for name in ("STOP_COMMAND.txt", "NARROW_COMMAND.txt", "CONTINUE_COMMAND.txt")]
    for path in command_paths:
        if path.exists():
            path.unlink()
    if verdict["classification"] == CONTINUE:
        (results_dir / "CONTINUE_COMMAND.txt").write_text(
            "CONTINUE_REVISED_FIBER_TRACK_TO_FINAL_GATE\n"
            "FIELD_DEFINING_STATUS=NOT_YET_ESTABLISHED\n"
            "NEXT_REQUIRED=independent_novelty_audit,independent_proof_review,specialized_system_baselines,calibrated_real_impairment\n"
            "STOP_IF_ANY_FINAL_GATE_FAILS=YES\n",
            encoding="utf-8",
        )
    elif verdict["classification"] == NARROW:
        (results_dir / "NARROW_COMMAND.txt").write_text(
            "NARROW_TO_FIXED_EDIT_FOUNDATIONS_STOP_FIELD_DEFINING\n"
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
    rows = e.get("largest_n_primary_rows", [])
    lines = [
        "# Revised FIBER-GRAND Decisive Gate",
        "",
        f"**Classification:** `{verdict['classification']}`",
        "",
        f"**Stage:** `{verdict['stage']}`",
        "",
        f"**Reason:** {verdict['reason']}",
        "",
        f"**Authorized follow-up:** {verdict['authorized_follow_up']}",
        "",
        "## Gate evidence",
        "",
        f"- Exactness: `{e['exactness_pass']}`",
        f"- Codebook-free syndrome/checksum operation: `{e['codebook_free_pass']}`",
        f"- Fixed-edit shell theorem: `{e['shell_theorem_pass']}`",
        f"- Strong syndrome-trellis baseline gate: `{e['strong_baseline_pass']}`",
        f"- VT specialized baseline: `{e['vt_specialized_pass']}`",
        f"- Two-deletion diagnostic: `{e['two_deletion_diagnostic_pass']}`",
        "",
        "## Largest-block primary comparison",
        "",
        "| Family | p | Trials | Fiber/trellis median wall | Fiber wall win | Agreement |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        ratio = float(row["median_fiber_over_best_wall"])
        lines.append(
            f"| {row['family']} | {float(row['p']):.3f} | {int(row['trials'])} | {ratio:.3f} | {str(ratio <= 0.8)} | {float(row['agreement_fraction']):.3f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "A computational pass is not a novelty judgment and does not establish real-system relevance. "
            "The final field-defining gate remains independent and mandatory.",
            "",
        ]
    )
    (results_dir / "GATE_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
