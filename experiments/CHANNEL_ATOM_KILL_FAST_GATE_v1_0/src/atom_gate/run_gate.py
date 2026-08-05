from __future__ import annotations

import argparse
import json
import shutil
import sys
import traceback
from pathlib import Path
from typing import Any

from .atlas import run_atlas
from .audit import run_exactness_audit
from .contract import decision_contract
from .decision import STOP, early_decision, final_decision, write_verdict_files
from .plots import generate_plots
from .rate_survival import run_rate_survival
from .scaling import run_nonlatin_scaling, run_reversible_action_scaling
from .utils import (
    build_sha256_manifest,
    ensure_dir,
    environment_record,
    seed_everything,
    write_json,
)


def load_profile(root: Path, profile: str) -> dict[str, Any]:
    profiles = json.loads((root / "config" / "profiles.json").read_text(encoding="utf-8"))
    if profile not in profiles:
        raise ValueError(f"Unknown profile {profile!r}; choose from {sorted(profiles)}")
    return profiles[profile]


def clean_results(results_dir: Path) -> None:
    ensure_dir(results_dir)
    for path in results_dir.iterdir():
        if path.name in {"README.md", ".gitkeep"}:
            continue
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()


def evidence_snapshot(
    exactness: dict[str, Any] | None,
    atlas: dict[str, Any] | None,
    reversible: dict[str, Any] | None,
    nonlatin: dict[str, Any] | None,
    rate: dict[str, Any] | None,
) -> dict[str, Any]:
    reversible_details: dict[str, Any] = {}
    if reversible is not None:
        for channel, row in reversible.get("channels", {}).items():
            reversible_details[channel] = {
                "exact_all": row.get("exact_all"),
                "largest_n_completion": row.get("largest_n_completion"),
                "largest_n_speedup": row.get("largest_n_speedup"),
                "atom_log2_work_slope": row.get("atom_log2_work_slope"),
                "reference_log2_work_slope": row.get("direct_log2_work_slope"),
                "slope_advantage": row.get("slope_advantage"),
            }

    nonlatin_details: dict[str, Any] = {}
    if nonlatin is not None:
        for channel, row in nonlatin.get("channel_best", {}).items():
            nonlatin_details[channel] = {
                "representation": row.get("representation"),
                "target_rate": row.get("target_rate"),
                "exact_all": row.get("exact_all"),
                "largest_n_completion": row.get("largest_n_completion"),
                "largest_n_speedup": row.get("largest_n_speedup"),
                "atom_slope": row.get("atom_slope"),
                "reference_slope": row.get("reference_slope"),
                "slope_advantage": row.get("slope_advantage"),
                "pessimistic_slope_advantage": row.get("pessimistic_slope_advantage"),
                "kappa": row.get("kappa"),
                "fiber_ceiling": row.get("fiber_ceiling"),
                "transition_degeneracy_bits": row.get("degeneracy"),
                "pass": nonlatin.get("channel_pass", {}).get(channel),
            }

    rate_details: dict[str, Any] = {}
    if rate is not None:
        for channel, row in rate.get("channel_best", {}).items():
            rate_details[channel] = {
                "best_representation": row.get("best_representation"),
                "uniform_input_mutual_information": row.get("uniform_input_mutual_information"),
                "fiber_ceiling_to_uniform_information": row.get("fiber_ceiling_to_uniform_information"),
                "atom_separating_rate_lower_bound": row.get("atom_separating_rate_lower_bound"),
                "rate_to_uniform_information": row.get("rate_to_uniform_information"),
                "target_mass": row.get("target_mass"),
                "actual_mass": row.get("actual_mass"),
                "mass_ok": row.get("mass_ok"),
                "milp_optimal": row.get("milp_optimal"),
                "pass": row.get("pass"),
            }

    return {
        "manual_novelty_gate_H0": "PENDING_INDEPENDENT_PRIMARY_SOURCE_AUDIT",
        "exactness_pass": None if exactness is None else exactness.get("pass"),
        "exactness_checks": None if exactness is None else exactness.get("count"),
        "atlas_witnesses": None if atlas is None else atlas.get("nonadditive_variation_witnesses"),
        "atlas_natural_improvements": None if atlas is None else atlas.get("nonadditive_natural_improvement_witnesses"),
        "atlas_channels": None if atlas is None else atlas.get("channels"),
        "atlas_representations": None if atlas is None else atlas.get("representations"),
        "reversible_pass": None if reversible is None else reversible.get("pass"),
        "reversible_details": reversible_details,
        "nonlatin_passing_channels": None if nonlatin is None else nonlatin.get("passing_channels"),
        "nonlatin_details": nonlatin_details,
        "rate_passing_channels": None if rate is None else rate.get("passing_channels"),
        "rate_details": rate_details,
    }


def finish(
    root: Path,
    results_dir: Path,
    verdict: dict[str, Any],
    exactness: dict[str, Any] | None,
    atlas: dict[str, Any] | None,
    reversible: dict[str, Any] | None,
    nonlatin: dict[str, Any] | None,
    rate: dict[str, Any] | None,
) -> int:
    evidence = evidence_snapshot(exactness, atlas, reversible, nonlatin, rate)
    write_verdict_files(results_dir, verdict, evidence)
    figures = generate_plots(results_dir)
    write_json(results_dir / "FIGURES.json", {"files": figures})
    build_sha256_manifest(results_dir, results_dir / "RESULTS_SHA256.txt")
    print("\n" + "=" * 78)
    print(f"FINAL CLASSIFICATION: {verdict['classification']}")
    print(f"STAGE: {verdict.get('stage', '')}")
    print(f"REASON: {verdict.get('reason', '')}")
    if verdict["classification"] == STOP:
        print("STOP COMMAND: STOP_BROAD_CHANNEL_ATOM_PROGRAM")
    elif verdict["classification"].startswith("PIVOT"):
        print("PIVOT COMMAND: STOP_ORIGINAL_BROAD_CHANNEL_ATOM_PROGRAM")
    else:
        print("CONTINUE COMMAND: CONTINUE_ONLY_TO_THEOREM_EXTRACTION_AND_INDEPENDENT_AUDIT")
    print("=" * 78)
    return 0


def run(profile_name: str) -> int:
    root = Path(__file__).resolve().parents[2]
    results_dir = root / "results"
    clean_results(results_dir)
    profile = load_profile(root, profile_name)
    rng = seed_everything(int(profile["seed"]))
    write_json(results_dir / "00_environment.json", environment_record())
    write_json(results_dir / "00_profile.json", {"profile": profile_name, "config": profile})
    write_json(results_dir / "00_decision_contract.json", decision_contract())

    exactness = atlas = reversible = nonlatin = rate = None
    try:
        print("[E0] Running exactness and theorem-implementation audit...")
        exactness = run_exactness_audit(results_dir, rng)
        write_json(results_dir / "RUN_STATUS.json", {"last_completed_stage": "E0", "pass": exactness["pass"]})
        verdict = early_decision(exactness=exactness)
        if verdict is not None:
            return finish(root, results_dir, verdict, exactness, atlas, reversible, nonlatin, rate)

        print("[E1] Building exact small-DMC representation atlas...")
        atlas = run_atlas(results_dir, rng, **profile["atlas"])
        write_json(results_dir / "RUN_STATUS.json", {"last_completed_stage": "E1", "pass": atlas["exact_all"]})
        verdict = early_decision(exactness=exactness, atlas=atlas)
        if verdict is not None:
            return finish(root, results_dir, verdict, exactness, atlas, reversible, nonlatin, rate)

        print("[E2] Running additive and non-cyclic reversible-action positive controls...")
        reversible = run_reversible_action_scaling(results_dir, rng, **profile["reversible"])
        write_json(results_dir / "RUN_STATUS.json", {"last_completed_stage": "E2", "pass": reversible["pass"]})
        verdict = early_decision(exactness=exactness, atlas=atlas, reversible=reversible)
        if verdict is not None:
            return finish(root, results_dir, verdict, exactness, atlas, reversible, nonlatin, rate)

        print("[E3] Running exact non-Latin block-scaling tests...")
        nonlatin = run_nonlatin_scaling(results_dir, rng, **profile["nonlatin"])
        write_json(results_dir / "RUN_STATUS.json", {"last_completed_stage": "E3", "pass": nonlatin["exact_all"]})
        if not nonlatin.get("exact_all", False):
            verdict = {
                "classification": STOP,
                "stage": "H1_BLOCK_EXACTNESS",
                "reason": "At least one block atom decoder disagreed with direct ML.",
                "authorized_follow_up": "Repair the exactness failure only; no scaling conclusion is valid.",
            }
            return finish(root, results_dir, verdict, exactness, atlas, reversible, nonlatin, rate)

        print("[E4] Running analytic and exact small-block rate-survival tests...")
        rate = run_rate_survival(results_dir, rng, **profile["rate_survival"])
        write_json(results_dir / "RUN_STATUS.json", {"last_completed_stage": "E4", "pass": True})

        verdict = final_decision(exactness, atlas, reversible, nonlatin, rate)
        return finish(root, results_dir, verdict, exactness, atlas, reversible, nonlatin, rate)

    except Exception as exc:
        tb = traceback.format_exc()
        (results_dir / "UNEXPECTED_FAILURE.txt").write_text(tb, encoding="utf-8")
        verdict = {
            "classification": STOP,
            "stage": "UNEXPECTED_IMPLEMENTATION_FAILURE",
            "reason": f"The computational gate terminated with an unhandled error: {exc!r}",
            "authorized_follow_up": "Repair and rerun the gate; no scientific performance conclusion is valid.",
        }
        return finish(root, results_dir, verdict, exactness, atlas, reversible, nonlatin, rate)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Channel-Atom Coding kill-fast scientific gate")
    parser.add_argument("--profile", choices=["smoke", "standard", "deep"], default="standard")
    args = parser.parse_args()
    raise SystemExit(run(args.profile))


if __name__ == "__main__":
    main()
