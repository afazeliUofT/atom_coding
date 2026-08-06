from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from .boundary import run_boundary_audit
from .code_transfer import run_code_transfer
from .contract import decision_contract, package_root, profiles
from .decision import final_decision, write_verdict_files
from .exactness import run_exactness_audit
from .insertion import run_one_insertion_scaling
from .one_deletion import run_one_deletion_scaling
from .plots import create_figures
from .theory_audit import run_theory_audit
from .two_deletion import run_two_deletion_scaling
from .utils import environment_snapshot, sha256_file, utc_now, write_json


def _copy_json(source: Path, destination: Path) -> dict[str, Any]:
    payload = json.loads(source.read_text(encoding="utf-8"))
    write_json(destination, payload)
    return payload


def _results_manifest(results_dir: Path) -> None:
    target = results_dir / "RESULTS_SHA256.txt"
    lines = []
    for path in sorted(results_dir.iterdir()):
        if path.is_file() and path.name != target.name:
            lines.append(f"{sha256_file(path)}  {path.name}")
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _evidence(
    exactness: dict[str, Any],
    theory: dict[str, Any],
    boundary: dict[str, Any],
    one_deletion: dict[str, Any],
    insertion: dict[str, Any],
    two_deletion: dict[str, Any],
    transfer: dict[str, Any],
) -> dict[str, Any]:
    boundary_rows = list(boundary.get("results", {}).values())
    one_rows = list(one_deletion.get("representations", {}).values())
    insertion_rows = list(insertion.get("representations", {}).values())
    two_rows = list(two_deletion.get("representations", {}).values())
    transfer_rows = list(transfer.get("results", {}).values())

    def best_largest(rows: list[dict[str, Any]], p: float) -> dict[str, Any] | None:
        candidates = [row for row in rows if abs(float(row.get("p", -1.0)) - p) <= 1e-12]
        if not candidates:
            return None
        return max(candidates, key=lambda row: float(row.get("largest_n", {}).get("mean_speedup", 0.0)))

    return {
        "exactness_checks": exactness.get("count"),
        "exactness_pass": exactness.get("pass"),
        "fiber_guesswork_identity": theory.get("fiber_guesswork_identity", {}).get("identity"),
        "pathwise_disagreement_configs": sum(
            row.get("pathwise_ml_disagreement_fraction", 0.0) >= 0.02 for row in boundary_rows
        ),
        "maximum_pathwise_disagreement_fraction": max(
            (row.get("pathwise_ml_disagreement_fraction", 0.0) for row in boundary_rows), default=0.0
        ),
        "maximum_median_search_inflation": max(
            (row.get("median_search_inflation_optimistic", 0.0) for row in boundary_rows), default=0.0
        ),
        "best_one_deletion_p005": None if best_largest(one_rows, 0.05) is None else best_largest(one_rows, 0.05)["largest_n"],
        "best_one_insertion_p005": None if best_largest(insertion_rows, 0.05) is None else best_largest(insertion_rows, 0.05)["largest_n"],
        "best_two_deletion_p005": None if best_largest(two_rows, 0.05) is None else best_largest(two_rows, 0.05)["largest_n"],
        "code_transfer_families": sorted(set(row.get("code_family") for row in transfer_rows)),
        "code_transfer_best_p005": {
            family: max(
                [row for row in transfer_rows if row.get("code_family") == family and abs(float(row.get("p", -1.0)) - 0.05) <= 1e-12],
                key=lambda row: (
                    float(row.get("mean_wall_speedup", 0.0)),
                    float(row.get("mean_speedup", 0.0)),
                ),
                default=None,
            )
            for family in sorted(set(row.get("code_family") for row in transfer_rows))
        },
    }


def run(profile_name: str, results_dir: Path | None = None) -> dict[str, Any]:
    root = package_root()
    all_profiles = profiles()
    if profile_name not in all_profiles:
        raise ValueError(f"Unknown profile {profile_name}")
    profile = all_profiles[profile_name]
    if profile.get("blocked_without_new_claim_contract"):
        raise RuntimeError(profile.get("reason", "Profile blocked"))

    results_dir = results_dir or (root / "results")
    results_dir.mkdir(parents=True, exist_ok=True)
    for path in results_dir.iterdir():
        if path.is_file() and path.name not in {"DEPENDENCY_INSTALL_LOG.txt", "LAUNCHER_ENVIRONMENT.json"}:
            path.unlink()

    write_json(results_dir / "00_environment.json", environment_snapshot())
    write_json(results_dir / "00_profile.json", {"profile": profile_name, "config": profile})
    write_json(results_dir / "00_decision_contract.json", decision_contract())
    rng = np.random.default_rng(int(profile["seed"]))
    start = time.perf_counter()

    print("[E0] Exact likelihood, certificate, and code-interface audit...", flush=True)
    exactness = run_exactness_audit(results_dir, rng)

    print("[E1] Theory/novelty-boundary identity audit...", flush=True)
    theory = run_theory_audit(results_dir)

    print("[E2] Path-versus-fiber boundary and search-inflation audit...", flush=True)
    boundary = run_boundary_audit(
        results_dir,
        rng,
        profile["boundary_blocklengths"],
        profile["boundary_probabilities"],
        int(profile["boundary_trials"]),
        int(profile["max_histories"]),
    )

    print("[E3] One-deletion-plus-substitution exact scaling...", flush=True)
    one_deletion = run_one_deletion_scaling(
        results_dir,
        rng,
        profile["one_deletion_blocklengths"],
        float(profile["one_deletion_rate"]),
        profile["one_deletion_probabilities"],
        profile["one_deletion_code_families"],
        profile["one_deletion_trials"],
        int(profile["max_histories"]),
        int(profile["max_prefix_nodes"]),
        int(profile["bootstrap_replicates"]),
    )

    print("[E4] One-insertion-plus-substitution transfer test...", flush=True)
    insertion = run_one_insertion_scaling(
        results_dir,
        rng,
        profile["insertion_blocklengths"],
        float(profile["insertion_rate"]),
        profile["insertion_probabilities"],
        profile["insertion_code_families"],
        profile["insertion_trials"],
        int(profile["max_histories"]),
        int(profile["bootstrap_replicates"]),
    )

    print("[E5] Two-deletion-plus-substitution fixed-edit test...", flush=True)
    two_deletion = run_two_deletion_scaling(
        results_dir,
        rng,
        profile["two_deletion_blocklengths"],
        float(profile["two_deletion_rate"]),
        profile["two_deletion_probabilities"],
        profile["two_deletion_code_families"],
        profile["two_deletion_trials"],
        int(profile["max_two_deletion_histories"]),
        int(profile["bootstrap_replicates"]),
    )

    print("[E6] Transfer across unmodified code families...", flush=True)
    transfer = run_code_transfer(
        results_dir,
        rng,
        profile["code_transfer_probabilities"],
        int(profile["code_transfer_trials"]),
        int(profile["max_histories"]),
        int(profile["max_prefix_nodes"]),
    )

    verdict, evaluations = final_decision(
        exactness,
        theory,
        boundary,
        one_deletion,
        insertion,
        two_deletion,
        transfer,
    )
    evidence = _evidence(exactness, theory, boundary, one_deletion, insertion, two_deletion, transfer)
    write_verdict_files(results_dir, verdict, evaluations, evidence)
    create_figures(results_dir)
    write_json(
        results_dir / "RUN_STATUS.json",
        {
            "pass": True,
            "last_completed_stage": "E6",
            "profile": profile_name,
            "classification": verdict["classification"],
            "utc_completed": utc_now(),
            "elapsed_seconds": time.perf_counter() - start,
        },
    )
    _results_manifest(results_dir)

    print("\n" + "=" * 78)
    print(f"FINAL CLASSIFICATION: {verdict['classification']}")
    print(f"STAGE: {verdict.get('stage','')}")
    print(f"REASON: {verdict.get('reason','')}")
    command_files = [name for name in ("STOP_COMMAND.txt", "PIVOT_COMMAND.txt", "CONTINUE_COMMAND.txt") if (results_dir / name).exists()]
    if command_files:
        print(f"COMMAND FILE: {command_files[0]}")
    print("=" * 78 + "\n")
    return {"verdict": verdict, "evaluations": evaluations, "evidence": evidence}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the FIBER-GRAND kill-fast scientific gate")
    parser.add_argument("--profile", default="standard", choices=["smoke", "standard", "deep"])
    parser.add_argument("--results-dir", type=Path, default=None)
    args = parser.parse_args(argv)
    try:
        run(args.profile, args.results_dir)
    except Exception as exc:
        print(f"EXECUTION_BLOCKED_NOT_A_SCIENTIFIC_VERDICT: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
