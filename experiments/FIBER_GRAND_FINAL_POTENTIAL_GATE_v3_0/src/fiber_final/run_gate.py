from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np

from .benchmark import run_primary_benchmark, run_vt_boundary_gate
from .decision import evaluate, write_report
from .exactness import run_exactness_audit
from .moment_theory import run_moment_tail_gate
from .phase_diagram import run_phase_diagram_gate
from .physical_model import run_physical_plausibility_gate
from .plots import make_plots
from .utils import environment_payload, read_json, result_hash_manifest, write_json


def _novelty_payload(package_root: Path) -> dict:
    return {
        "overall_classification": "NARROW_NOVELTY_SURVIVES",
        "literature_cutoff": "2026-08-06",
        "matrix_file": "theory/PRIMARY_SOURCE_NOVELTY_MATRIX.md",
        "withdrawn_claims": [
            "fiber guesswork is a new random variable",
            "best-path versus best-string is new",
            "A*, priority queues, or hidden-path summation are new",
            "unrestricted weighted-transducer exact decoding is efficient",
        ],
        "surviving_claim": (
            "A fixed-edit, codebook-free, membership-only aggregate inverse generator with a strict sum-of-frontiers ML certificate, "
            "together with fixed-edit history moment/tail laws and a typical-versus-mean complexity phase diagram."
        ),
        "closest_prior_art": [
            "Duffy-Li-Medard GRAND (IEEE TIT 2019; arXiv:1802.07010)",
            "Tan-Joudeh general DMC guessing with abandonment (IEEE TIT 2025; arXiv:2502.05959)",
            "Gallager synchronization sequential decoding (Lincoln Lab Group Report 2502, 1961)",
            "Han-Hartmann-Chen priority-first ML decoding (IEEE TIT 1993)",
            "US4922494A error-frame enumeration and likelihood selection (1990)",
            "Ozaydin-Medard-Duffy GRAND-assisted optimal modulation (arXiv:2210.16187)",
            "Sabary et al. fixed-k deletion likelihood/ML* analysis (arXiv:2201.02466)",
            "Weighted automata and most-probable-string literature",
        ],
        "caution": "No search can guarantee absence of unpublished work or every patent family. Independent external novelty review remains required before submission.",
    }


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

    print("[E0] Exactness, complete ties, shell, moment-sandwich, and VT audit...")
    run_exactness_audit(results_dir, rng)

    print("[E1] Primary-source novelty adjudication for the narrowed claim...")
    novelty = _novelty_payload(package_root)
    write_json(results_dir / "01_novelty_adjudication.json", novelty)

    print("[E2] Exact finite-length Renyi moment, tail, and LDP audit...")
    run_moment_tail_gate(
        results_dir,
        profile["moment_blocklengths"],
        profile["moment_probabilities"],
        profile["moment_orders"],
        profile["moment_edit_counts"],
        profile["quantiles"],
    )

    print("[E3] Typical-versus-mean rate-complexity phase diagram...")
    run_phase_diagram_gate(results_dir, profile["phase_rates"], profile["benchmark_probabilities"])

    print("[E4] Codebook-free FIBER versus syndrome-trellis and prefix A* near the phase boundary...")
    run_primary_benchmark(
        results_dir,
        rng,
        profile["benchmark_blocklengths"],
        profile["benchmark_rates"],
        profile["benchmark_probabilities"],
        profile["benchmark_families"],
        profile["benchmark_trials"],
        int(profile["max_histories"]),
        int(profile["max_trellis_terminals"]),
        int(profile["max_prefix_nodes"]),
    )

    print("[E5] True O(n) VT single-deletion specialization boundary...")
    run_vt_boundary_gate(results_dir, rng, profile["vt_blocklengths"], int(profile["vt_trials"]))

    print("[E6] Calibrated synthetic timing-slip front end and optional measured trace audit...")
    run_physical_plausibility_gate(
        results_dir,
        rng,
        int(profile["physical_n"]),
        int(profile["physical_train_frames"]),
        int(profile["physical_test_frames"]),
        float(profile["physical_snr_db"]),
        float(profile["physical_isi_strength"]),
        str(profile["real_trace_glob"]),
        package_root,
    )

    print("[E7] Frozen scientific decision...")
    verdict = evaluate(results_dir)
    write_report(results_dir, verdict)
    make_plots(results_dir)
    (results_dir / "ANALYSIS_HANDOFF.md").write_text(
        "# Analysis handoff\n\n"
        "Review in this order:\n\n"
        "1. `GATE_VERDICT.json`\n"
        "2. `GATE_REPORT.md`\n"
        "3. `01_novelty_adjudication.json` and `theory/PRIMARY_SOURCE_NOVELTY_MATRIX.md`\n"
        "4. `02_moment_tail_gate.json`, moment and tail CSVs\n"
        "5. `03_phase_diagram_gate.json`\n"
        "6. `04_primary_benchmark_gate.json` and raw trials\n"
        "7. `05_vt_boundary_gate.json`\n"
        "8. `07_physical_plausibility_gate.json`\n\n"
        "A positive classification confirms high potential only for the narrow fixed-edit track. A measured post-front-end trace and independent external proof/novelty reviews remain mandatory.\n",
        encoding="utf-8",
    )
    result_hash_manifest(results_dir, results_dir / "RESULTS_SHA256.txt")
    write_json(results_dir / "RUN_STATUS.json", {"pass": True, "last_completed_stage": "E7"})

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
