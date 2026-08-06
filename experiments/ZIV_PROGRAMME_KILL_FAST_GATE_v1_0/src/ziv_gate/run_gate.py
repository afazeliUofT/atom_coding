from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np

from .audit import run_exactness_audit
from .code_geometry import run_code_geometry_gate
from .contract import decision_contract, package_root, profiles
from .decision import final_decision, write_verdict_files
from .individual_sequence import run_individual_sequence_gate
from .masking import run_masking_gate
from .plots import make_plots
from .rank_benchmark import run_nonstationary_adaptation_benchmark, run_stationary_rank_benchmark
from .trace_audit import run_trace_gate
from .utils import environment_snapshot, sha256_manifest, write_json


def run(profile_name: str, output_dir: Path | None = None) -> dict[str, Any]:
    root = package_root()
    profile_table = profiles()
    if profile_name not in profile_table:
        raise ValueError(f"Unknown profile {profile_name!r}")
    profile = profile_table[profile_name]
    if profile_name == "deep" and profile.get("blocked_by_default", False):
        raise RuntimeError(
            "The deep profile is blocked by default. It may be enabled only after the standard gate emits a "
            "scientifically justified surviving claim contract; do not use more computation to rescue a failed hypothesis."
        )
    results = root / "results" if output_dir is None else output_dir
    results.mkdir(parents=True, exist_ok=True)
    # Preserve README only when using the package results directory.
    for path in list(results.iterdir()):
        if path.name == "README.md":
            continue
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()

    rng = np.random.default_rng(int(profile["seed"]))
    write_json(results / "00_environment.json", environment_snapshot())
    write_json(results / "00_profile.json", {"profile": profile_name, "config": profile})
    write_json(results / "00_decision_contract.json", decision_contract())

    print("[E0] Exactness, normalization, and theorem-implementation audit...", flush=True)
    exactness = run_exactness_audit(results, rng)
    if not exactness.get("pass", False):
        verdict = final_decision(exactness, {}, {}, {}, {}, {}, {})
        evidence = {"exactness_pass": False, "exactness_checks": exactness.get("count", 0)}
        write_verdict_files(results, verdict, evidence)
        write_json(results / "RUN_STATUS.json", {"pass": False, "last_completed_stage": "E0"})
        sha256_manifest(results, results / "RESULTS_SHA256.txt")
        return verdict

    print("[E1] Individual-sequence theorem, tie, ensemble, and regret gates...", flush=True)
    individual = run_individual_sequence_gate(results, profile["individual_blocklengths"])

    print("[E2] Exact and adversarial XOR masking-defect search...", flush=True)
    masking = run_masking_gate(
        results,
        rng,
        profile["masking_exact_blocklengths"],
        profile["masking_heuristic_blocklengths"],
        profile["ctw_depths"],
        int(profile["masking_random_count"]),
        int(profile["masking_random_pairs"]),
        int(profile["masking_hill_steps"]),
    )

    print("[E3] Stationary finite-state universal-rank benchmark...", flush=True)
    stationary = run_stationary_rank_benchmark(
        results,
        rng,
        profile["stationary_blocklengths"],
        profile["stationary_rates"],
        int(profile["stationary_trials"]),
        profile["training_lengths"],
    )

    print("[E4] Nonstationary online-adaptation benchmark...", flush=True)
    nonstationary = run_nonstationary_adaptation_benchmark(
        results,
        rng,
        int(profile["adaptation_n"]),
        float(profile["adaptation_rate"]),
        int(profile["adaptation_frames"]),
        int(profile["adaptation_switch_every"]),
        int(profile["adaptation_sessions"]),
        float(profile["adaptation_discount"]),
    )

    print("[E5] Finite-block description-length code-geometry tests...", flush=True)
    geometry = run_code_geometry_gate(
        results,
        rng,
        profile["geometry_blocklengths"],
        profile["geometry_rates"],
        int(profile["geometry_replicates"]),
        int(profile["geometry_candidate_pool"]),
        int(profile["geometry_ctw_depth"]),
        int(profile["geometry_ctw_max_n"]),
    )

    print("[E6] Synthetic entropy and optional real-trace audit...", flush=True)
    traces = run_trace_gate(
        results,
        root,
        profile["stationary_rates"],
        int(profile["geometry_ctw_depth"]),
        int(profile["trace_max_symbols"]),
    )

    verdict = final_decision(exactness, individual, masking, stationary, nonstationary, geometry, traces)
    ctw_violation = any(
        bool(row.get("exact_any_positive_violation") or row.get("heuristic_any_positive_violation"))
        for metric, row in masking.get("combined", {}).items()
        if metric.startswith("CTW")
    )
    evidence = {
        "exactness_pass": exactness.get("pass", False),
        "exactness_checks": exactness.get("count", 0),
        "tie_gap_per_symbol": individual["original_deterministic_conjecture"]["counterexample"]["gap_per_symbol"],
        "repaired_rank_regret": individual["repaired_level_set_atlas"]["largest_n_worst_normalized_regret"],
        "lz_mask_survives": masking.get("combined", {}).get("LZ78_FIXED_BLOCK", {}).get("survives_falsification_only", False),
        "ctw_mask_violation": ctw_violation,
        "stationary_pass": stationary.get("pass", False),
        "stationary_regimes": stationary.get("structured_passing_regimes", 0),
        "adaptation_pass": nonstationary.get("pass", False),
        "adaptation_gain": nonstationary.get("discounted_adaptive_gain_over_fixed_fit_per_symbol"),
        "geometry_pass": geometry.get("pass", False),
        "geometry_signal": geometry.get("positive_structured_code_geometry_signal", False),
        "traces_pass": traces.get("pass", False),
        "trace_status": traces.get("manual_systems_gate", "unknown"),
    }
    write_verdict_files(results, verdict, evidence)
    make_plots(results)
    write_json(results / "RUN_STATUS.json", {"pass": True, "last_completed_stage": "E6"})
    sha256_manifest(results, results / "RESULTS_SHA256.txt")

    print("\n" + "=" * 78)
    print(f"FINAL CLASSIFICATION: {verdict['classification']}")
    print(f"STAGE: {verdict.get('stage','')}")
    print(f"REASON: {verdict.get('reason','')}")
    if verdict["classification"].startswith("STOP"):
        print("STOP COMMAND: STOP_FIELD_DEFINING_ZIV_PROGRAMME")
    elif verdict["classification"].startswith("PIVOT"):
        print("PIVOT COMMAND: STOP_ORIGINAL_FOUR_PILLAR_ZIV_PROGRAMME")
    else:
        print("CONTINUE COMMAND: CONTINUE_ONLY_TO_THEOREM_EXTRACTION_AND_INDEPENDENT_NOVELTY_AUDIT")
    print("=" * 78 + "\n")
    return verdict


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=sorted(profiles()), default="standard")
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args(argv)
    run(args.profile, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
