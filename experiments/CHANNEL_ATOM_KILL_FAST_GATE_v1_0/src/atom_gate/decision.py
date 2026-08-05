from __future__ import annotations

from pathlib import Path
from typing import Any

from .contract import decision_contract
from .utils import write_json


CONTINUE = "CONTINUE_FIELD_DEFINING_TRACK"
PIVOT = "PIVOT_TO_REVISED_TRACK"
STOP = "STOP_BROAD_CHANNEL_ATOM_PROGRAM"


def early_decision(
    exactness: dict[str, Any] | None = None,
    atlas: dict[str, Any] | None = None,
    reversible: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    if exactness is not None and not exactness.get("pass", False):
        return {
            "classification": STOP,
            "stage": "H1_EXACTNESS",
            "reason": "At least one exactness, decomposition, fiber-oracle, or enumeration audit failed.",
            "authorized_follow_up": "Repair or verify the failed audit only; no performance work is authorized.",
        }
    if atlas is not None:
        thresholds = decision_contract()["atlas_H2"]
        mechanism_pass = bool(
            atlas.get("exact_all", False)
            and atlas.get("nonadditive_variation_witnesses", 0)
            >= int(thresholds["minimum_nonadditive_variation_witnesses"])
            and atlas.get("nonadditive_natural_improvement_witnesses", 0)
            >= int(thresholds["minimum_nonadditive_natural_improvement_witnesses"])
        )
        if not mechanism_pass:
            return {
                "classification": STOP,
                "stage": "H2_OPERATIONAL_REPRESENTATION_DEPENDENCE",
                "reason": (
                    "The exact small-DMC atlas did not show a sufficiently broad reduced representation effect "
                    "or did not improve over a natural independent coupling."
                ),
                "authorized_follow_up": "Prepare a boundary/negative result; do not begin block theory.",
            }
    if reversible is not None and not reversible.get("pass", False):
        return {
            "classification": STOP,
            "stage": "REVERSIBLE_ACTION_POSITIVE_CONTROL",
            "reason": (
                "The exact first-hit positive controls did not produce reliable exact scaling gains over direct ML. "
                "The broader non-Latin programme has no credible computational base."
            ),
            "authorized_follow_up": "Audit implementation and cost model once; if confirmed, terminate the programme.",
        }
    return None


def final_decision(
    exactness: dict[str, Any],
    atlas: dict[str, Any],
    reversible: dict[str, Any],
    nonlatin: dict[str, Any],
    rate_survival: dict[str, Any],
) -> dict[str, Any]:
    early = early_decision(exactness, atlas, reversible)
    if early is not None:
        return early

    nonlatin_pass = {
        channel for channel, passed in nonlatin.get("channel_pass", {}).items() if passed
    }
    rate_pass = {
        channel for channel, passed in rate_survival.get("channel_pass", {}).items() if passed
    }
    joint = sorted(nonlatin_pass & rate_pass)

    minimum_joint = int(
        decision_contract()["nonlatin_H3"]["minimum_jointly_passing_channel_families_for_continue"]
    )
    if len(joint) >= minimum_joint:
        return {
            "classification": CONTINUE,
            "stage": "H3_H4_PASSED_PROVISIONALLY",
            "reason": (
                "At least two structurally distinct non-Latin channel families show exact, fully costed favorable "
                "scaling and pass the pre-registered rate-survival diagnostics."
            ),
            "joint_passing_channels": joint,
            "authorized_follow_up": (
                "Proceed to theorem extraction: prove a strict work separation, develop compact multiplicity aggregation, "
                "and construct an explicit positive-rate atom-separating code family."
            ),
            "warning": (
                "This is evidence of potential, not a field-defining result or proof of asymptotic superiority. "
                "The manual H0 novelty gate remains pending."
            ),
        }

    if len(joint) == 1:
        channel = joint[0]
        return {
            "classification": PIVOT,
            "stage": "TARGETED_NONADDITIVE_FAMILY_ONLY",
            "reason": (
                f"Only {channel} jointly passed scaling and rate survival. The broad channel-universal claim is not supported."
            ),
            "surviving_path": f"Restrict the programme to {channel} and derive a channel-specific representation/fiber oracle.",
            "authorized_follow_up": "Freeze a new narrow claim contract and rerun a family-specific kill gate.",
            "stop_original_claim": True,
        }

    if nonlatin_pass:
        channels = sorted(nonlatin_pass)
        return {
            "classification": PIVOT,
            "stage": "DECODER_GAIN_WITHOUT_RATE_SURVIVAL",
            "reason": (
                "Some non-Latin decoder scaling signals exist, but no channel jointly passes the rate-survival gate."
            ),
            "surviving_path": (
                "A finite-length or decoder-only result may remain for: " + ", ".join(channels)
            ),
            "authorized_follow_up": "Stop code-design claims; test a decoder-only application with fixed deployed codes.",
            "stop_original_claim": True,
        }

    noncyclic_pass = bool(
        reversible.get("pass_flags", {}).get("NONCYCLIC_REVERSIBLE_ACTION_q5", False)
    )
    if noncyclic_pass:
        return {
            "classification": PIVOT,
            "stage": "REVERSIBLE_ACTION_ONLY",
            "reason": (
                "The broad non-Latin programme failed, but the non-cyclic bi-unambiguous action channel passed the exact "
                "first-hit scaling control."
            ),
            "surviving_path": (
                "Stop Channel-Atom Coding as a general exact-ML framework and evaluate a narrower Random Reversible Action GRAND theory."
            ),
            "authorized_follow_up": (
                "First prove that the action class is not a relabelled/quasigroup restatement of known GRAND and identify "
                "a real channel where it provides an implementation advantage."
            ),
            "stop_original_claim": True,
        }

    return {
        "classification": STOP,
        "stage": "H3_H4_FAILED",
        "reason": (
            "No non-Latin channel jointly shows favorable fully accounted scaling and rate survival, and no defensible "
            "restricted action path remains."
        ),
        "authorized_follow_up": "Archive results and prepare a negative/boundary analysis only.",
    }


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.{digits}g}"
    return str(value)


def write_verdict_files(results_dir: Path, verdict: dict[str, Any], evidence: dict[str, Any]) -> None:
    payload = {**verdict, "evidence_summary": evidence}
    write_json(results_dir / "GATE_VERDICT.json", payload)
    classification = verdict["classification"]

    lines = [
        "# Channel-Atom Kill-Fast Verdict",
        "",
        f"**Classification:** `{classification}`",
        "",
        f"**Decision stage:** `{verdict.get('stage', 'unknown')}`",
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
            "## Decisive evidence",
            "",
            "| Gate | Result | Key evidence |",
            "|---|---:|---|",
            (
                f"| H0 novelty | pending | Independent primary-source audit required; numerical success cannot close H0. |"
            ),
            (
                f"| H1 exactness | {_fmt(evidence.get('exactness_pass'))} | "
                f"{_fmt(evidence.get('exactness_checks'))} independent implementation checks. |"
            ),
            (
                f"| H2 one-letter representation effect | "
                f"{_fmt((evidence.get('atlas_witnesses') or 0) > 0)} | "
                f"{_fmt(evidence.get('atlas_witnesses'))} nonadditive variation witnesses and "
                f"{_fmt(evidence.get('atlas_natural_improvements'))} improvements over the natural independent coupling "
                f"across {_fmt(evidence.get('atlas_channels'))} channels / {_fmt(evidence.get('atlas_representations'))} reduced representations. |"
            ),
            (
                f"| Reversible-action positive control | {_fmt(evidence.get('reversible_pass'))} | "
                f"See the exact first-hit metrics below. |"
            ),
            (
                f"| H3 non-Latin exact scaling | {_fmt((evidence.get('nonlatin_passing_channels') or 0) > 0)} | "
                f"{_fmt(evidence.get('nonlatin_passing_channels'))} channel families passed the frozen scaling thresholds. |"
            ),
            (
                f"| H4 rate survival | {_fmt((evidence.get('rate_passing_channels') or 0) > 0)} | "
                f"{_fmt(evidence.get('rate_passing_channels'))} channel families passed both fiber and likely-atom rate diagnostics. |"
            ),
        ]
    )

    reversible_details = evidence.get("reversible_details", {})
    if reversible_details:
        lines.extend(
            [
                "",
                "### Reversible-action positive controls",
                "",
                "| Channel | Completion at largest n | Largest-n speedup | Atom slope | Reference slope | Slope advantage |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for channel, row in sorted(reversible_details.items()):
            lines.append(
                f"| `{channel}` | {_fmt(row.get('largest_n_completion'))} | "
                f"{_fmt(row.get('largest_n_speedup'))} | {_fmt(row.get('atom_log2_work_slope'))} | "
                f"{_fmt(row.get('reference_log2_work_slope'))} | {_fmt(row.get('slope_advantage'))} |"
            )

    nonlatin_details = evidence.get("nonlatin_details", {})
    if nonlatin_details:
        lines.extend(
            [
                "",
                "### Best non-Latin result found per channel family",
                "",
                "| Channel | Rate | Completion at largest n | Largest-n speedup | Atom slope | Exact-reference slope | Advantage | Pessimistic advantage | Pass |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for channel, row in sorted(nonlatin_details.items()):
            lines.append(
                f"| `{channel}` | {_fmt(row.get('target_rate'))} | {_fmt(row.get('largest_n_completion'))} | "
                f"{_fmt(row.get('largest_n_speedup'))} | {_fmt(row.get('atom_slope'))} | "
                f"{_fmt(row.get('reference_slope'))} | {_fmt(row.get('slope_advantage'))} | "
                f"{_fmt(row.get('pessimistic_slope_advantage'))} | {_fmt(row.get('pass'))} |"
            )

    rate_details = evidence.get("rate_details", {})
    if rate_details:
        lines.extend(
            [
                "",
                "### Rate-survival diagnostics",
                "",
                "| Channel | Fiber ceiling / uniform information | Atom-separating rate / uniform information | Target mass | Actual mass | Exact MILP | Pass |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for channel, row in sorted(rate_details.items()):
            lines.append(
                f"| `{channel}` | {_fmt(row.get('fiber_ceiling_to_uniform_information'))} | "
                f"{_fmt(row.get('rate_to_uniform_information'))} | {_fmt(row.get('target_mass'))} | "
                f"{_fmt(row.get('actual_mass'))} | {_fmt(row.get('milp_optimal'))} | {_fmt(row.get('pass'))} |"
            )

    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "This is a pre-registered research-investment classification from finite exact and numerical tests. "
            "It is not a universal impossibility proof and not a declaration that a field-defining result already exists. "
            "The H0 novelty gate requires independent manual review of primary literature.",
            "",
        ]
    )
    (results_dir / "GATE_REPORT.md").write_text("\n".join(lines), encoding="utf-8")

    handoff = [
        "# Analysis Handoff",
        "",
        "Use this file to review the run without searching the repository.",
        "",
        "## Read in this order",
        "",
        "1. `GATE_VERDICT.json` — final machine classification and compact evidence.",
        "2. `GATE_REPORT.md` — numerical interpretation and stop/pivot boundary.",
        "3. `01_exactness_audit.json` — independent correctness checks.",
        "4. `02_small_dmc_atlas_summary.json` and `02_small_dmc_channel_summary.csv` — H2.",
        "5. `04_reversible_action_scaling.json` and `04_reversible_action_summary.csv` — positive controls.",
        "6. `05_nonlatin_scaling.json` and `05_nonlatin_summary.csv` — decisive H3 tests.",
        "7. `06_rate_survival.json`, `06_rate_survival_analytic.csv`, and `06_rate_survival_graph.csv` — H4.",
        "8. `RESULTS_SHA256.txt` — integrity manifest.",
        "",
        "Exactly one of `STOP_COMMAND.txt`, `PIVOT_COMMAND.txt`, or `CONTINUE_COMMAND.txt` is authoritative for investment discipline.",
        "",
        "The detailed compressed trial tables are retained so that means, tails, completion fractions, and fitted slopes can be recomputed independently.",
        "",
    ]
    (results_dir / "ANALYSIS_HANDOFF.md").write_text("\n".join(handoff), encoding="utf-8")

    stop_path = results_dir / "STOP_COMMAND.txt"
    pivot_path = results_dir / "PIVOT_COMMAND.txt"
    continue_path = results_dir / "CONTINUE_COMMAND.txt"
    for path in (stop_path, pivot_path, continue_path):
        if path.exists():
            path.unlink()

    if classification == STOP:
        stop_path.write_text(
            "STOP_BROAD_CHANNEL_ATOM_PROGRAM\n"
            f"STAGE={verdict.get('stage', '')}\n"
            f"REASON={verdict.get('reason', '')}\n"
            "DO_NOT_START=asymptotic_theory,code_family_design,finite_state_extensions,hardware,applications\n",
            encoding="utf-8",
        )
    elif classification == PIVOT:
        pivot_path.write_text(
            "STOP_ORIGINAL_BROAD_CHANNEL_ATOM_PROGRAM\n"
            f"PIVOT_STAGE={verdict.get('stage', '')}\n"
            f"SURVIVING_PATH={verdict.get('surviving_path', '')}\n"
            "REQUIRE_NEW_CLAIM_CONTRACT=YES\n",
            encoding="utf-8",
        )
    else:
        continue_path.write_text(
            "CONTINUE_ONLY_TO_THEOREM_EXTRACTION_AND_INDEPENDENT_AUDIT\n"
            "FIELD_DEFINING_STATUS=NOT_YET_ESTABLISHED\n",
            encoding="utf-8",
        )
