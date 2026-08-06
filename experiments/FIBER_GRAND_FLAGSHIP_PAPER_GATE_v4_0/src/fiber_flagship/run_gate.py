from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np

from .benchmark import run_vt_boundary_gate
from .candidate_theory import run_candidate_theory_gate
from .compiled_gate import run_compiled_flagship_gate
from .decision import evaluate, write_report
from .exactness import run_exactness_audit
from .moment_theory import run_moment_tail_gate
from .phase_diagram import run_phase_diagram_gate
from .plots import make_plots
from .trace_gate import run_measured_trace_gate
from .utils import environment_payload, read_json, result_hash_manifest, write_json


def run(profile_name: str) -> dict:
    package_root = Path(__file__).resolve().parents[2]
    results_dir = package_root / "results"
    results_dir.mkdir(exist_ok=True)
    for path in results_dir.iterdir():
        if path.is_file():
            path.unlink()
        elif path.is_dir():
            shutil.rmtree(path)

    profiles = read_json(package_root / "config" / "profiles.json")
    if profile_name not in profiles:
        raise ValueError(f"unknown profile {profile_name}")
    profile = profiles[profile_name]
    contract = read_json(package_root / "config" / "decision_contract.json")
    write_json(results_dir / "00_profile.json", {"profile": profile_name, "config": profile})
    write_json(results_dir / "00_decision_contract.json", contract)
    write_json(results_dir / "00_environment.json", environment_payload())
    rng = np.random.default_rng(int(profile["seed"]))

    print("[E0] Exactness, complete ties, certificates, moment sandwich, and VT audit...")
    run_exactness_audit(results_dir, rng)

    print("[E1] Candidate-volume, ambiguity, and L0 membership-oracle competitiveness audit...")
    run_candidate_theory_gate(
        results_dir,
        rng,
        profile["candidate_exhaustive_n"],
        int(profile["candidate_random_trials"]),
    )

    print("[E2] Fixed-edit moment/tail theory and typical-versus-mean phase diagram...")
    run_moment_tail_gate(
        results_dir,
        profile["moment_blocklengths"],
        profile["moment_probabilities"],
        profile["moment_orders"],
        profile["moment_edit_counts"],
        profile["quantiles"],
    )
    run_phase_diagram_gate(results_dir, profile["phase_rates"], sorted({p for row in profile["compiled_schedules"] for p in row["probabilities"]}))

    print("[E3] C++20 FIBER versus equally compiled exact prefix A* benchmark...")
    run_compiled_flagship_gate(package_root, results_dir, profile, contract, rng)

    print("[E4] Optional measured post-front-end trace audit...")
    run_measured_trace_gate(package_root, results_dir, str(profile["real_trace_glob"]))

    print("[E5] True O(n) VT specialization boundary...")
    run_vt_boundary_gate(results_dir, rng, profile["vt_blocklengths"], int(profile["vt_trials"]))

    print("[E6] Frozen flagship-paper readiness decision...")
    verdict = evaluate(results_dir)
    write_report(results_dir, verdict)
    make_plots(results_dir)
    (results_dir / "ANALYSIS_HANDOFF.md").write_text(
        "# Analysis handoff\n\n"
        "Review in this order:\n\n"
        "1. `GATE_VERDICT.json` and `GATE_REPORT.md`;\n"
        "2. `01_candidate_theory_gate.json` and the theory PDF;\n"
        "3. `02_moment_tail_gate.json` and `03_phase_diagram_gate.json`;\n"
        "4. `03_compiled_gate.json`, summary, slopes, and raw trials;\n"
        "5. `05_vt_boundary_gate.json`;\n"
        "6. `04_measured_trace_gate.json`;\n"
        "7. `theory/EXTERNAL_REVIEW_HANDOFF.md`.\n\n"
        "A positive result authorizes a full IEEE TIT manuscript and external review. It does not establish field-defining impact or real-system value.\n",
        encoding="utf-8",
    )
    result_hash_manifest(results_dir, results_dir / "RESULTS_SHA256.txt")
    write_json(results_dir / "RUN_STATUS.json", {"pass": True, "last_completed_stage": "E6"})

    print("\n" + "=" * 78)
    print(f"FINAL CLASSIFICATION: {verdict['classification']}")
    print(f"STAGE: {verdict['stage']}")
    print(f"REASON: {verdict['reason']}")
    print("=" * 78 + "\n")
    return verdict


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=("smoke", "standard"), default="standard")
    args = parser.parse_args()
    run(args.profile)


if __name__ == "__main__":
    main()
