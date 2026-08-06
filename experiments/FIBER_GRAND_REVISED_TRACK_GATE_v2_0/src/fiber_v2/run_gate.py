from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

import numpy as np

from .benchmark import (
    run_primary_strong_baseline,
    run_shell_theorem_gate,
    run_two_deletion_diagnostic,
    run_vt_specialized_gate,
)
from .decision import evaluate, write_report
from .exactness import run_exactness_audit
from .plots import make_plots
from .utils import environment_payload, read_json, result_hash_manifest, write_json


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=["smoke", "standard", "deep"], default="standard")
    args = parser.parse_args()
    package_root = Path(__file__).resolve().parents[2]
    profiles = read_json(package_root / "config" / "profiles.json")
    if args.profile == "deep":
        print("DEEP_PROFILE_BLOCKED: a passed standard gate and new claim contract are required.")
        return 7
    config = profiles[args.profile]
    results_dir = package_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    for path in results_dir.iterdir():
        if path.is_file():
            path.unlink()
        elif path.is_dir():
            shutil.rmtree(path)
    write_json(results_dir / "00_environment.json", environment_payload())
    write_json(results_dir / "00_profile.json", {"profile": args.profile, "config": config})
    contract = read_json(package_root / "config" / "decision_contract.json")
    write_json(results_dir / "00_decision_contract.json", contract)
    rng = np.random.default_rng(int(config["seed"]))
    start = time.perf_counter()

    print("[E0] Independent exactness and no-codebook oracle audit...", flush=True)
    exactness = run_exactness_audit(results_dir, rng)
    if not exactness["pass"]:
        verdict = {
            "classification": "STOP_FIBER_FIELD_DEFINING_PROGRAMME",
            "stage": "EXACTNESS_AUDIT_FAILURE",
            "reason": "At least one independent exactness or certificate audit failed.",
            "authorized_follow_up": "Repair only a demonstrable implementation defect; no performance run is authorized.",
            "evidence_summary": {"exactness_pass": False},
        }
        write_json(results_dir / "GATE_VERDICT.json", verdict)
        (results_dir / "STOP_COMMAND.txt").write_text("STOP_FIBER_FIELD_DEFINING_PROGRAMME\n", encoding="utf-8")
        (results_dir / "GATE_REPORT.md").write_text(
            "# Revised FIBER-GRAND Decisive Gate\n\n**Classification:** `STOP_FIBER_FIELD_DEFINING_PROGRAMME`\n\nExactness audit failed; no performance conclusion is authorized.\n",
            encoding="utf-8",
        )
        write_json(results_dir / "RUN_STATUS.json", {"pass": False, "last_completed_stage": "E0"})
        result_hash_manifest(results_dir, results_dir / "RESULTS_SHA256.txt")
        return 2

    print("[E1] Membership-only FIBER versus exact syndrome-trellis aggregate search...", flush=True)
    run_primary_strong_baseline(
        results_dir,
        rng,
        config["primary_blocklengths"],
        float(config["primary_rate"]),
        config["primary_probabilities"],
        config["primary_families"],
        config["primary_trials"],
        int(config["max_histories"]),
        int(config["max_trellis_terminals"]),
        int(config["max_prefix_nodes"]),
    )

    print("[E2] Two-deletion codebook-free diagnostic...", flush=True)
    run_two_deletion_diagnostic(
        results_dir,
        rng,
        config["two_deletion_blocklengths"],
        float(config["primary_rate"]),
        config["two_deletion_probabilities"],
        config["two_deletion_trials"],
        int(config["max_two_deletion_histories"]),
    )

    print("[E3] Specialized VT one-deletion baseline...", flush=True)
    run_vt_specialized_gate(
        results_dir,
        rng,
        config["vt_blocklengths"],
        int(config["vt_trials"]),
    )

    print("[E4] Fixed-edit shell-certificate theorem audit...", flush=True)
    run_shell_theorem_gate(
        results_dir,
        rng,
        config["theorem_blocklengths"],
        config["theorem_probabilities"],
        int(config["theorem_samples"]),
    )

    verdict = evaluate(results_dir)
    write_report(results_dir, verdict)
    make_plots(results_dir)
    write_json(results_dir / "RUN_STATUS.json", {"pass": True, "last_completed_stage": "E4", "elapsed_seconds": time.perf_counter() - start})
    (results_dir / "ANALYSIS_HANDOFF.md").write_text(
        "# Analysis handoff\n\nRead: GATE_VERDICT.json, GATE_REPORT.md, 02_primary_gate.json, 05_shell_theorem_gate.json, then CSV trial files.\n",
        encoding="utf-8",
    )
    result_hash_manifest(results_dir, results_dir / "RESULTS_SHA256.txt")

    print("\n" + "=" * 78)
    print(f"FINAL CLASSIFICATION: {verdict['classification']}")
    print(f"STAGE: {verdict['stage']}")
    print(f"REASON: {verdict['reason']}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
