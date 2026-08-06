from __future__ import annotations

from pathlib import Path
from typing import Any

from .contract import decision_contract
from .utils import write_json


CONTINUE = "CONTINUE_FIELD_DEFINING_ZIV_TRACK"
PIVOT = "PIVOT_TO_REVISED_ZIV_TRACK"
STOP = "STOP_FIELD_DEFINING_ZIV_PROGRAMME"


def final_decision(
    exactness: dict[str, Any],
    individual: dict[str, Any],
    masking: dict[str, Any],
    stationary: dict[str, Any],
    nonstationary: dict[str, Any],
    geometry: dict[str, Any],
    traces: dict[str, Any],
) -> dict[str, Any]:
    if not exactness.get("pass", False):
        return {
            "classification": STOP,
            "stage": "IMPLEMENTATION_AUDIT",
            "reason": "At least one exactness, normalization, type-counting, or code audit failed.",
            "authorized_follow_up": "Repair and independently verify the failed implementation only; no scientific performance conclusion is authorized.",
        }

    original_failed = not individual["original_deterministic_conjecture"].get("pass", False)
    regret_failed = not individual["session_regret_transfer"].get("pass_as_stated", False)
    lz_mask = masking.get("combined", {}).get("LZ78_FIXED_BLOCK", {})
    lz_survives = bool(lz_mask.get("survives_falsification_only", False))
    geometry_pass = bool(geometry.get("pass", False))
    stationary_pass = bool(stationary.get("pass", False))
    adaptation_pass = bool(nonstationary.get("pass", False))
    traces_pass = bool(traces.get("pass", False))

    # Full continuation requires a new foundational pillar plus practical evidence and real traces.
    if (
        not original_failed
        and not regret_failed
        and lz_survives
        and geometry_pass
        and stationary_pass
        and traces_pass
    ):
        return {
            "classification": CONTINUE,
            "stage": "ALL_EARLY_NECESSARY_GATES_PASSED",
            "reason": "The original theorem, effective geometry, finite-block mechanism, and real-trace premise all survived the frozen gate.",
            "authorized_follow_up": "Proceed only to theorem extraction and an independent primary-source novelty audit; field-defining status is not yet established.",
        }

    # A geometry-only pivot is authorized only by both masking survival and positive code behavior.
    if lz_survives and geometry_pass:
        return {
            "classification": PIVOT,
            "stage": "DESCRIPTION_LENGTH_GEOMETRY_ONLY",
            "reason": (
                "The deterministic individual-sequence and signed-regret claims failed, but the frozen LZ78 geometry survived "
                "counterexample search and produced a structured-error code-design signal."
            ),
            "stop_original_claim": True,
            "surviving_path": (
                "Freeze a narrow theorem contract for one explicit LZ78 codelength: prove or refute the XOR masking inequality, "
                "then derive a nontrivial positive-rate code family."
            ),
            "authorized_follow_up": "No soft decoder, hardware, or application work until that theorem gate closes.",
        }

    # An adaptive systems result may remain publishable, but is not a field-defining replacement by itself.
    if stationary_pass and adaptation_pass:
        return {
            "classification": PIVOT,
            "stage": "ADAPTIVE_UNIVERSAL_GRAND_SYSTEMS_ONLY",
            "reason": (
                "Synthetic finite-block ranking supports an adaptive universal-GRAND implementation, but the flagship deterministic theorem "
                "and signed-regret transfer fail, the stochastic finite-state core is already established, and no code geometry passed."
            ),
            "stop_original_claim": True,
            "surviving_path": (
                "A narrower engineering programme on online-adaptive universal GRAND under abrupt model drift may be tested against fitted HMM/GRAND-MO baselines. "
                "It is not presently a credible field-defining coding theory."
            ),
            "authorized_follow_up": (
                "Only a newly frozen application-specific claim contract and real post-front-end traces may authorize further work; "
                "do not continue the original four-pillar programme."
            ),
        }

    return {
        "classification": STOP,
        "stage": "FLAGSHIP_AND_GEOMETRY_COLLAPSE",
        "reason": (
            "The original deterministic individual-sequence theorem is structurally false, the session regret implication is false as stated, "
            "and the frozen practical description-length geometry did not produce a viable finite-block code-design signal. "
            "Any stochastic universal-ranking gains are within an already established research line."
        ),
        "authorized_follow_up": "Archive the gate and prepare only a negative/boundary analysis. Do not run the deep profile or begin soft/hardware/application work.",
    }


def _fmt(value: Any, digits: int = 4) -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.{digits}g}"
    return str(value)


def write_verdict_files(results_dir: Path, verdict: dict[str, Any], evidence: dict[str, Any]) -> None:
    payload = {**verdict, "evidence_summary": evidence, "manual_novelty_gate": "PENDING_INDEPENDENT_PRIMARY_SOURCE_ADJUDICATION"}
    write_json(results_dir / "GATE_VERDICT.json", payload)
    classification = verdict["classification"]
    lines = [
        "# Ziv Programme Kill-Fast Verdict",
        "",
        f"**Classification:** `{classification}`",
        "",
        f"**Decision stage:** `{verdict.get('stage','')}`",
        "",
        f"**Reason:** {verdict.get('reason','')}",
        "",
        f"**Authorized follow-up:** {verdict.get('authorized_follow_up','')}",
    ]
    if verdict.get("surviving_path"):
        lines.extend(["", f"**Surviving path:** {verdict['surviving_path']}"])
    lines.extend(
        [
            "",
            "## Gate evidence",
            "",
            "| Gate | Result | Evidence |",
            "|---|---:|---|",
            f"| Z0 implementation | {_fmt(evidence['exactness_pass'])} | {_fmt(evidence['exactness_checks'])} independent audit groups. |",
            f"| Z1 deterministic individual-sequence theorem | no | Exact one-state uniform-tie counterexample; normalized gap {_fmt(evidence['tie_gap_per_symbol'])}. |",
            f"| Z1R repaired level-set finite atlas | diagnostic | Largest-n worst normalized LZ rank regret {_fmt(evidence['repaired_rank_regret'])}. |",
            f"| Z2 session signed-regret transfer | no | Cancellation counterexample reproduced. |",
            f"| Z3 LZ masking counterexample search | {_fmt(evidence['lz_mask_survives'])} | Survives falsification only; this is not a theorem. |",
            f"| Z3 CTW masking | {_fmt(not evidence['ctw_mask_violation'])} | Positive practical CTW violation found: {_fmt(evidence['ctw_mask_violation'])}. |",
            f"| Z4 stationary universal-rank signal | {_fmt(evidence['stationary_pass'])} | Structured regimes passing: {_fmt(evidence['stationary_regimes'])}. |",
            f"| Z5 nonstationary adaptation | {_fmt(evidence['adaptation_pass'])} | Discounted adaptive gain/symbol {_fmt(evidence['adaptation_gain'])}. |",
            f"| Z6 code geometry | {_fmt(evidence['geometry_pass'])} | Structured LZ-only correction signal {_fmt(evidence['geometry_signal'])}. |",
            f"| Z7 real-trace premise | {_fmt(evidence['traces_pass'])} | {evidence['trace_status']}. |",
            "| Z8 novelty | pending | Requires independent primary-source audit; known stochastic and randomized-individual-sequence results bound the claim. |",
            "",
            "## Interpretation boundary",
            "",
            "The finite computations can refute universal statements and reveal finite-block mechanisms, but cannot prove an asymptotic masking theorem or establish novelty. ",
            "A PIVOT classification stops the original four-pillar programme and authorizes only the explicitly stated narrow claim contract.",
            "",
        ]
    )
    (results_dir / "GATE_REPORT.md").write_text("\n".join(lines), encoding="utf-8")

    for name in ("STOP_COMMAND.txt", "PIVOT_COMMAND.txt", "CONTINUE_COMMAND.txt"):
        path = results_dir / name
        if path.exists():
            path.unlink()
    if classification == STOP:
        (results_dir / "STOP_COMMAND.txt").write_text(
            "STOP_FIELD_DEFINING_ZIV_PROGRAMME\n"
            f"STAGE={verdict.get('stage','')}\n"
            f"REASON={verdict.get('reason','')}\n"
            "DO_NOT_START=deep_profile,full_theory,soft_decoder,hardware,real_world_trials,code_family_programme\n",
            encoding="utf-8",
        )
    elif classification == PIVOT:
        (results_dir / "PIVOT_COMMAND.txt").write_text(
            "STOP_ORIGINAL_FOUR_PILLAR_ZIV_PROGRAMME\n"
            f"PIVOT_STAGE={verdict.get('stage','')}\n"
            f"SURVIVING_PATH={verdict.get('surviving_path','')}\n"
            "REQUIRE_NEW_CLAIM_CONTRACT=YES\n",
            encoding="utf-8",
        )
    else:
        (results_dir / "CONTINUE_COMMAND.txt").write_text(
            "CONTINUE_ONLY_TO_THEOREM_EXTRACTION_AND_INDEPENDENT_NOVELTY_AUDIT\n"
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
        "4. `02_individual_sequence_gate.json` and repaired-rank CSV files",
        "5. `03_masking_gate.json` and masking witness CSV files",
        "6. `04_stationary_rank_gate.json`",
        "7. `05_nonstationary_gate.json`",
        "8. `06_code_geometry_gate.json`",
        "9. `07_trace_gate.json`",
        "10. `RESULTS_SHA256.txt`",
        "",
        "Exactly one command file is authoritative for investment discipline.",
        "",
    ]
    (results_dir / "ANALYSIS_HANDOFF.md").write_text("\n".join(handoff), encoding="utf-8")
