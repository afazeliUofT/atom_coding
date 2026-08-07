from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import binom

PACKAGE_NAME = "FIBER_GRAND_V5_E3_TAIL_ADJUDICATION_R1"
SOURCE_NAME = "FIBER_GRAND_PRE_MANUSCRIPT_CLOSURE_v5_0"
EXPECTED_ROWS = {0: 960, 1: 960, 2: 3600, 3: 3600}
EXPECTED_FAILURES = {
    ("random_linear", 0.005, 561),
    ("random_linear", 0.010, 31),
    ("named_crc_linear", 0.005, 382),
    ("named_crc_linear", 0.010, 105),
    ("named_crc_linear", 0.010, 185),
    ("named_crc_linear", 0.010, 296),
    ("named_crc_linear", 0.010, 314),
}
SHELL3_CAP = 5_341_184


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def slope_log2(x: pd.Series, y: pd.Series) -> float:
    xv = np.asarray(x, dtype=float)
    yv = np.asarray(y, dtype=float)
    if len(xv) < 2 or np.any(yv <= 0) or not np.all(np.isfinite(yv)):
        return float("nan")
    return float(np.polyfit(xv, np.log2(yv), 1)[0])


def bootstrap_median_ci(values: np.ndarray, rng: np.random.Generator, reps: int = 2000) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    median = float(np.median(values))
    if len(values) < 2:
        return median, median, median
    draws = rng.choice(values, size=(reps, len(values)), replace=True)
    medians = np.median(draws, axis=1)
    return median, float(np.quantile(medians, 0.025)), float(np.quantile(medians, 0.975))


def distribution_free_quantile_ci(values: np.ndarray, quantile: float, confidence: float = 0.95) -> tuple[float, float, float]:
    values = np.sort(np.asarray(values, dtype=float))
    n = len(values)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    estimate = float(np.quantile(values, quantile))
    alpha = 1.0 - confidence
    lo_count = int(binom.ppf(alpha / 2.0, n, quantile))
    hi_count = int(binom.ppf(1.0 - alpha / 2.0, n, quantile))
    lo = max(0, min(n - 1, lo_count - 1))
    hi = max(0, min(n - 1, hi_count))
    return estimate, float(values[lo]), float(values[hi])


def error_bucket(weight: int) -> str:
    if weight <= 0:
        return "E0"
    if weight == 1:
        return "E1"
    if weight == 2:
        return "E2"
    return "E3PLUS"


def h2(p: float) -> float:
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -p * math.log2(p) - (1.0 - p) * math.log2(1.0 - p)


def hhalf(p: float) -> float:
    return 2.0 * math.log2(math.sqrt(p) + math.sqrt(1.0 - p))


def threshold(target: float, kind: str) -> float:
    lo, hi = 0.0, 0.5
    for _ in range(100):
        mid = (lo + hi) / 2.0
        val = h2(mid) if kind == "typical" else hhalf(mid)
        if val < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def run_command(command: list[str], cwd: Path, log: Path, allow_codes: set[int] = {0}) -> subprocess.CompletedProcess[str]:
    started = time.perf_counter()
    proc = subprocess.run(
        command,
        cwd=str(cwd),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
        env={**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
    )
    with log.open("a", encoding="utf-8") as f:
        f.write(f"$ {' '.join(command)}\n{proc.stdout}\nELAPSED_SECONDS={time.perf_counter()-started:.6f}\nRETURN_CODE={proc.returncode}\n")
    if proc.returncode not in allow_codes:
        raise RuntimeError(f"command failed ({proc.returncode}): {' '.join(command)}\n{proc.stdout}")
    return proc


def load_original(source_results: Path) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    frames: list[pd.DataFrame] = []
    inventory: list[dict[str, Any]] = []
    for index, expected in EXPECTED_ROWS.items():
        path = source_results / f"03_compiled_trials_part{index:02d}.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path)
        if len(frame) != expected:
            raise RuntimeError(f"{path.name}: expected {expected} rows, found {len(frame)}")
        frames.append(frame)
        inventory.append({"file": path.name, "rows": len(frame), "sha256": sha256(path)})
    full = pd.concat(frames, ignore_index=True)
    failures = full[(full["agreement"] != 1) | (full["fiber_certified"] != 1) | ((full["prefix_certified"] != 1) & (full["trellis_certified"] != 1))]
    keys = {(str(r.family), round(float(r.p), 6), int(r.trial)) for r in failures.itertuples()}
    if keys != EXPECTED_FAILURES:
        raise RuntimeError(f"unexpected original failure keys: {sorted(keys)}")
    if not (failures["error_weight"] == 3).all():
        raise RuntimeError("not all original failures are E=3")
    return full, inventory


def compile_and_replay(package_root: Path, source_root: Path, results_dir: Path, timing_repeats: int) -> pd.DataFrame:
    build = package_root / "build"
    build.mkdir(exist_ok=True)
    binary = build / "replay_failures"
    source = package_root / "cpp" / "replay_failures.cpp"
    log = results_dir / "01_replay_build_and_run.log"
    log.write_text("", encoding="utf-8")
    compiler = shutil.which("g++")
    if not compiler:
        raise RuntimeError("g++ is required")
    run_command([compiler, "-O3", "-DNDEBUG", "-std=c++20", "-march=native", "-Wall", "-Wextra", str(source), "-o", str(binary)], package_root, log)
    output = results_dir / "01_e3_failure_replay.csv"
    run_command([str(binary), str(output), str(timing_repeats)], package_root, log)
    replay = pd.read_csv(output)
    keys = {(str(r.family), round(float(r.p), 6), int(r.trial)) for r in replay.itertuples()}
    if keys != EXPECTED_FAILURES or len(replay) != len(EXPECTED_FAILURES):
        raise RuntimeError(f"replay keys mismatch: {sorted(keys)}")
    if not (replay["error_weight"] == 3).all():
        raise RuntimeError("replay did not reproduce E=3 for every target")
    if not (replay["shell3_history_cap"] == SHELL3_CAP).all():
        raise RuntimeError("wrong shell-complete history cap")
    return replay


def patch_rows(original: pd.DataFrame, replay: pd.DataFrame) -> pd.DataFrame:
    frame = original.copy()
    for row in replay.itertuples(index=False):
        mask = (
            (frame["family"] == row.family)
            & np.isclose(frame["p"], float(row.p))
            & (frame["trial"] == int(row.trial))
            & (frame["n"] == 64)
            & np.isclose(frame["rate"], 0.75)
        )
        if int(mask.sum()) != 1:
            raise RuntimeError(f"target row not unique: {row.family}, {row.p}, {row.trial}")
        original_row = frame.loc[mask].iloc[0]
        if int(original_row["error_weight"]) != int(row.error_weight) or int(original_row["deleted_position"]) != int(row.deleted_position):
            raise RuntimeError("deterministic replay metadata mismatch")
        updates = {
            "agreement": int(row.agreement),
            "fiber_certified": int(row.fiber_certified),
            "prefix_certified": int(row.prefix_certified),
            "fiber_wall": float(row.fiber_wall),
            "prefix_wall": float(row.prefix_wall),
            "trellis_wall": 0.0,
            "best_baseline_wall": float(row.prefix_wall) if int(row.prefix_certified) else float("inf"),
            "best_baseline_name": "PREFIX_ASTAR",
            "fiber_over_prefix_wall": float(row.fiber_over_prefix_wall),
            "fiber_over_best_wall": float(row.fiber_over_prefix_wall) if int(row.prefix_certified) else float("inf"),
            "fiber_histories": int(row.fiber_histories),
            "fiber_candidates": int(row.fiber_candidates),
            "fiber_duplicates": int(row.fiber_duplicates),
            "fiber_membership": int(row.fiber_membership),
            "fiber_exact_scores": int(row.fiber_exact_scores),
            "fiber_peak_seen": int(row.fiber_peak_seen),
            "fiber_peak_frontier": int(row.fiber_peak_frontier),
            "prefix_nodes": int(row.prefix_nodes),
            "prefix_membership": int(row.prefix_membership),
            "prefix_exact_scores": int(row.prefix_exact_scores),
            "prefix_peak_frontier": int(row.prefix_peak_frontier),
        }
        for key, value in updates.items():
            frame.loc[mask, key] = value
    return frame


def summarize(frame: pd.DataFrame, contract: dict[str, Any], rng: np.random.Generator, results_dir: Path) -> dict[str, Any]:
    frame = frame.copy()
    frame["error_bucket"] = frame["error_weight"].map(error_bucket)
    frame.to_csv(results_dir / "02_repaired_compiled_trials.csv.gz", index=False, compression="gzip")
    summary_rows: list[dict[str, Any]] = []
    for (family, rate, n, p), group in frame.groupby(["family", "rate", "n", "p"]):
        ratio = group["fiber_over_best_wall"].to_numpy(float)
        med, lo, hi = bootstrap_median_ci(ratio, rng)
        p99, p99_lo, p99_hi = distribution_free_quantile_ci(ratio, 0.99)
        redundancy = 1.0 - float(rate)
        p_typ = threshold(redundancy, "typical")
        p_mean = threshold(redundancy, "mean")
        summary_rows.append({
            "family": family, "rate": float(rate), "n": int(n), "p": float(p), "trials": int(len(group)),
            "positive_error_trials": int((group["error_weight"] >= 1).sum()),
            "agreement_fraction": float(group["agreement"].mean()),
            "completion_fraction": float((group["fiber_certified"].astype(bool) & (group["prefix_certified"].astype(bool) | group["trellis_certified"].astype(bool))).mean()),
            "median_fiber_over_best_wall": med, "median_ratio_ci95_low": lo, "median_ratio_ci95_high": hi,
            "p90_fiber_over_best_wall": float(np.quantile(ratio, .90)), "p95_fiber_over_best_wall": float(np.quantile(ratio, .95)),
            "p99_fiber_over_best_wall": p99, "p99_ratio_ci95_low": p99_lo, "p99_ratio_ci95_high": p99_hi,
            "maximum_fiber_over_best_wall": float(np.max(ratio)),
            "median_fiber_wall": float(group["fiber_wall"].median()), "median_prefix_wall": float(group["prefix_wall"].median()),
            "median_trellis_wall": float(group.loc[group["trellis_certified"] == 1, "trellis_wall"].median()) if (group["trellis_certified"] == 1).any() else float("nan"),
            "median_best_baseline_wall": float(group["best_baseline_wall"].median()),
            "prefix_best_fraction": float((group["best_baseline_name"] == "PREFIX_ASTAR").mean()),
            "trellis_best_fraction": float((group["best_baseline_name"] == "SYNDROME_TRELLIS").mean()),
            "trellis_available_fraction": float(group["trellis_available"].mean()),
            "trellis_certified_fraction": float(group["trellis_certified"].mean()),
            "trellis_estimated_updates": int(group["trellis_estimated_updates"].iloc[0]),
            "median_histories": float(group["fiber_histories"].median()), "p95_histories": float(group["fiber_histories"].quantile(.95)),
            "p99_histories": float(group["fiber_histories"].quantile(.99)), "median_membership_queries": float(group["fiber_membership"].median()),
            "median_prefix_nodes": float(group["prefix_nodes"].median()), "typical_threshold_p": p_typ, "mean_threshold_p": p_mean,
            "predicted_typical_favorable": bool(float(p) < p_typ), "predicted_mean_favorable": bool(float(p) < p_mean),
        })
    summary = pd.DataFrame(summary_rows).sort_values(["family", "rate", "p", "n"])
    summary.to_csv(results_dir / "02_repaired_compiled_summary.csv", index=False)

    error_rows: list[dict[str, Any]] = []
    for (family, rate, n, p, bucket), group in frame.groupby(["family", "rate", "n", "p", "error_bucket"]):
        ratio = group["fiber_over_best_wall"].to_numpy(float)
        error_rows.append({
            "family": family, "rate": float(rate), "n": int(n), "p": float(p), "error_bucket": bucket,
            "trials": int(len(group)), "agreement_fraction": float(group["agreement"].mean()),
            "completion_fraction": float((group["fiber_certified"].astype(bool) & (group["prefix_certified"].astype(bool) | group["trellis_certified"].astype(bool))).mean()),
            "median_fiber_over_best_wall": float(np.median(ratio)), "p90_fiber_over_best_wall": float(np.quantile(ratio,.90)),
            "p95_fiber_over_best_wall": float(np.quantile(ratio,.95)), "p99_fiber_over_best_wall": float(np.quantile(ratio,.99)),
            "maximum_fiber_over_best_wall": float(np.max(ratio)), "median_histories": float(group["fiber_histories"].median()),
            "p95_histories": float(group["fiber_histories"].quantile(.95)), "trellis_available_fraction": float(group["trellis_available"].mean()),
            "trellis_best_fraction": float((group["best_baseline_name"] == "SYNDROME_TRELLIS").mean()),
        })
    error_summary = pd.DataFrame(error_rows).sort_values(["family", "rate", "p", "n", "error_bucket"])
    error_summary.to_csv(results_dir / "02_repaired_error_weight_summary.csv", index=False)

    slope_rows: list[dict[str, Any]] = []
    for (family, rate, p), group in summary.groupby(["family", "rate", "p"]):
        group = group.sort_values("n")
        if len(group) < 2:
            continue
        availability = group["trellis_available_fraction"].to_numpy(float)
        slope_rows.append({
            "family": family, "rate": float(rate), "p": float(p),
            "fiber_wall_log2_slope": slope_log2(group["n"], group["median_fiber_wall"]),
            "best_baseline_wall_log2_slope": slope_log2(group["n"], group["median_best_baseline_wall"]),
            "fiber_minus_best_slope": slope_log2(group["n"], group["median_fiber_wall"]) - slope_log2(group["n"], group["median_best_baseline_wall"]),
            "ratio_log2_slope": slope_log2(group["n"], group["median_fiber_over_best_wall"]),
            "trellis_available_all_lengths": bool(np.all(availability >= 1-1e-12)),
            "reference_policy_consistent": bool(np.all(availability >= 1-1e-12) or np.all(availability <= 1e-12)),
            "predicted_typical_favorable": bool(group["predicted_typical_favorable"].iloc[0]),
            "predicted_mean_favorable": bool(group["predicted_mean_favorable"].iloc[0]),
        })
    slopes = pd.DataFrame(slope_rows)
    slopes.to_csv(results_dir / "02_repaired_compiled_slopes.csv", index=False)

    c = contract["compiled_benchmark"]
    largest_n = int(summary["n"].max())
    largest = summary[summary["n"] == largest_n]
    key = largest[largest["predicted_typical_favorable"] & (largest["p"] <= float(c["maximum_key_probability"]))]
    mean_favorable = key[key["predicted_mean_favorable"]]
    positive = error_summary[(error_summary["n"] == largest_n) & (error_summary["error_bucket"] != "E0") & (error_summary["p"] <= float(c["maximum_key_probability"])) & (error_summary["trials"] >= int(c["minimum_positive_error_stratum_trials"]))]

    statuses = []
    trellis_coverage = True
    required_rows: list[dict[str, Any]] = []
    for req in c["required_trellis_points"]:
        subset = summary[(summary["n"] == int(req["n"])) & np.isclose(summary["rate"], float(req["rate"])) & (summary["p"] <= float(req.get("maximum_p", c["maximum_key_probability"])))]
        rows = subset.to_dict(orient="records")
        required_rows.extend(rows)
        passed = bool(rows and all(float(r["trellis_available_fraction"]) >= 1.0 and float(r["trellis_certified_fraction"]) >= 1.0 for r in rows))
        statuses.append({**req, "matched_rows": len(rows), "pass": passed})
        trellis_coverage &= passed

    median_wins = int((key["median_fiber_over_best_wall"] <= float(c["median_win_ratio"])).sum())
    robust_ci = int((key["median_ratio_ci95_high"] <= float(c["maximum_median_ci_high"])).sum())
    p99_wins = int((key["p99_fiber_over_best_wall"] <= float(c["maximum_p99_ratio"])).sum())
    p99_ci = int((key["p99_ratio_ci95_high"] <= float(c["maximum_p99_ci_high"])).sum())
    mean_p99 = bool(not mean_favorable.empty and float(mean_favorable["p99_fiber_over_best_wall"].max()) <= float(c["maximum_mean_favorable_p99_ratio"]))
    consistent_slopes = slopes[slopes["predicted_typical_favorable"] & slopes["reference_policy_consistent"]]
    slope_passes = int((consistent_slopes["fiber_minus_best_slope"] <= float(c["maximum_slope_disadvantage"])).sum())
    pos_passes = int(((positive["median_fiber_over_best_wall"] <= float(c["positive_error_median_ratio"])) & (positive["p95_fiber_over_best_wall"] <= float(c["positive_error_p95_ratio"]))).sum())
    min_trials = bool(not key.empty and int(key["trials"].min()) >= int(c["minimum_key_trials_per_config"]))
    pos_evidence = bool(pos_passes >= int(c["minimum_positive_error_strata_passes"]))
    exact_all = bool(frame["agreement"].all())
    completion_all = bool((frame["fiber_certified"].astype(bool) & (frame["prefix_certified"].astype(bool) | frame["trellis_certified"].astype(bool))).all())
    pass_flag = bool(
        exact_all and completion_all and trellis_coverage and min_trials
        and len(key) >= int(c["minimum_key_largest_n_configs"])
        and median_wins >= int(c["minimum_median_wins"])
        and robust_ci >= int(c["minimum_robust_ci_wins"])
        and p99_wins >= int(c["minimum_p99_wins"])
        and p99_ci >= int(c["minimum_p99_ci_wins"])
        and mean_p99 and slope_passes >= int(c["minimum_slope_passes"])
        and pos_evidence
    )
    payload = {
        "pass": pass_flag, "exact_agreement_all": exact_all, "completion_all": completion_all,
        "rows": int(len(frame)), "largest_n": largest_n, "key_largest_n_rows": key.to_dict(orient="records"),
        "positive_error_rows": positive.to_dict(orient="records"), "required_trellis_rows": required_rows,
        "trellis_requirement_status": statuses, "trellis_coverage_pass": trellis_coverage,
        "minimum_key_trials_pass": min_trials, "median_win_count": median_wins, "robust_ci_win_count": robust_ci,
        "p99_win_count": p99_wins, "p99_ci_win_count": p99_ci, "mean_favorable_p99_pass": mean_p99,
        "slope_pass_count": slope_passes, "positive_error_pass_count": pos_passes, "positive_error_evidence_pass": pos_evidence,
        "interpretation": "The original v5 dataset is retained unchanged. Only the seven censored E=3 rows are deterministically replayed with the theorem-derived complete-through-shell-3 history cap; this adjudication cannot retroactively make the original 2M-cap gate pass.",
    }
    write_json(results_dir / "02_repaired_compiled_gate.json", payload)
    return payload


def run_vt_boundary(source_root: Path, results_dir: Path) -> dict[str, Any]:
    sys.path.insert(0, str(source_root / "src"))
    from fiber_flagship.benchmark import run_vt_boundary_gate
    rng = np.random.default_rng(20260807)
    return run_vt_boundary_gate(results_dir, rng, [15,31,63,127], 20)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--timing-repeats", type=int, default=3)
    args = parser.parse_args()
    repo = Path(args.repo_root).resolve()
    package_root = Path(__file__).resolve().parents[1]
    source_root = repo / "experiments" / SOURCE_NAME
    source_results = source_root / "results"
    results_dir = package_root / "results"
    if not source_root.exists():
        raise FileNotFoundError(source_root)
    results_dir.mkdir(exist_ok=True)
    for p in results_dir.iterdir():
        if p.is_file() or p.is_symlink(): p.unlink()
        elif p.is_dir(): shutil.rmtree(p)

    original, inventory = load_original(source_results)
    original_failure_rows = original[(original["agreement"] != 1) | (original["fiber_certified"] != 1) | ((original["prefix_certified"] != 1) & (original["trellis_certified"] != 1))]
    original_failure_rows.to_csv(results_dir / "00_original_v5_failures.csv", index=False)
    write_json(results_dir / "00_input_inventory.json", {"source_commit": "29e18ba2a18c6fb3f9c230311e326e74514ca3ab", "files": inventory, "original_rows": len(original), "original_failure_count": len(original_failure_rows), "all_failures_E3": bool((original_failure_rows["error_weight"] == 3).all())})

    replay = compile_and_replay(package_root, source_root, results_dir, args.timing_repeats)
    replay_pass = bool(replay["fiber_certified"].all() and replay["prefix_certified"].all() and replay["agreement"].all())
    write_json(results_dir / "01_replay_gate.json", {
        "pass": replay_pass, "targets": len(replay), "all_E3": bool((replay["error_weight"] == 3).all()),
        "all_fiber_certified_by_shell3": bool(replay["fiber_certified"].all()),
        "all_prefix_certified_at_10M": bool(replay["prefix_certified"].all()),
        "all_complete_ties_agree": bool(replay["agreement"].all()),
        "maximum_fiber_histories": int(replay["fiber_histories"].max()),
        "shell3_history_cap": SHELL3_CAP,
        "maximum_fiber_over_prefix_wall": float(replay["fiber_over_prefix_wall"].max()),
        "median_fiber_over_prefix_wall": float(replay["fiber_over_prefix_wall"].median()),
    })

    repaired = patch_rows(original, replay)
    contract = read_json(source_results / "00_decision_contract.json")
    compiled = summarize(repaired, contract, np.random.default_rng(20260807), results_dir)
    vt = run_vt_boundary(source_root, results_dir)
    exact = read_json(source_results / "00_exactness_audit.json")
    candidate = read_json(source_results / "01_candidate_theory_gate.json")
    moment = read_json(source_results / "02_moment_tail_gate.json")
    phase = read_json(source_results / "03_phase_diagram_gate.json")
    foundational_pass = bool(exact.get("pass") and candidate.get("candidate_volume_bounds_pass") and candidate.get("exact_ambiguity_degree_pass") and candidate.get("oracle_to_fiber_inflation_bounds_pass") and candidate.get("known_cardinality_refinement_pass") and phase.get("mean_threshold_below_typical_all") and moment.get("gap_decrease_fraction",0) >= .8)
    vt_pass = bool(vt.get("agreement_all") and vt.get("linear_candidate_check_identity"))

    if not foundational_pass or not bool(replay["fiber_certified"].all()) or not bool(replay["agreement"].all()):
        classification = "STOP_FLAGSHIP_PERFORMANCE_CLAIM"
        reason = "A foundational audit or the theorem-derived shell-complete replay failed."
    elif compiled.get("pass") and vt_pass and bool(replay["prefix_certified"].all()):
        classification = "AUTHORIZE_FULL_TIT_MANUSCRIPT_AND_EXTERNAL_REVIEW"
        reason = "The seven v5 E=3 censored rows certified under the theorem-derived shell-complete cap, agreed with the extended exact prefix baseline, and the repaired full dataset passed the frozen statistical contract."
    else:
        classification = "NARROW_TO_FIXED_EDIT_TIT_THEORY_ALGORITHM_PAPER"
        reason = "The theory and exact decoder survive, but shell-complete replay or the repaired full performance contract does not support the flagship performance claim."

    verdict = {
        "classification": classification,
        "field_defining_status": "NOT_ESTABLISHED",
        "original_v5_frozen_status": "FAILED_2M_RESOURCE_CAP_ON_7_E3_ROWS",
        "original_v5_retroactive_pass_permitted": False,
        "reason": reason,
        "foundational_theory_pass": foundational_pass,
        "replay_pass": replay_pass,
        "repaired_compiled_contract_pass": bool(compiled.get("pass")),
        "vt_boundary_pass": vt_pass,
        "stop_if_external_review_finds_blocking_defect": True,
    }
    write_json(results_dir / "GATE_VERDICT.json", verdict)
    lines = [
        "# FIBER-GRAND v5 E=3 Tail Adjudication R1", "",
        f"**Classification:** `{classification}`", "",
        f"**Reason:** {reason}", "",
        "## Immutable interpretation", "",
        "Commit `29e18ba` remains a failed 2,000,000-history-cap run; this adjudication does not rewrite it.",
        "The seven failures were all E=3 resource-cap censoring events. The replay uses the theorem-derived full-through-shell-3 cap of 5,341,184 histories and a 10,000,000-node exact prefix reference.", "",
        f"- Foundational theory: `{foundational_pass}`",
        f"- All seven FIBER shell-complete certifications: `{bool(replay['fiber_certified'].all())}`",
        f"- All seven exact-prefix certifications: `{bool(replay['prefix_certified'].all())}`",
        f"- Complete tie agreement: `{bool(replay['agreement'].all())}`",
        f"- Repaired full statistical contract: `{bool(compiled.get('pass'))}`",
        f"- VT boundary: `{vt_pass}`", "",
    ]
    (results_dir / "GATE_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    if classification == "AUTHORIZE_FULL_TIT_MANUSCRIPT_AND_EXTERNAL_REVIEW":
        (results_dir / "CONTINUE_COMMAND.txt").write_text("AUTHORIZE_FULL_TIT_MANUSCRIPT_AND_EXTERNAL_REVIEW\nORIGINAL_V5_CAP_FAILURE_MUST_BE_REPORTED=YES\nNEXT_REQUIRED=full_TIT_manuscript,independent_proof_review,independent_novelty_review\n", encoding="utf-8")
    elif classification == "NARROW_TO_FIXED_EDIT_TIT_THEORY_ALGORITHM_PAPER":
        (results_dir / "NARROW_COMMAND.txt").write_text("NARROW_TO_FIXED_EDIT_TIT_THEORY_ALGORITHM_PAPER\nFLAGSHIP_PERFORMANCE_CLAIM=STOP\nNO_FURTHER_PERFORMANCE_RESCUE_GATES=YES\n", encoding="utf-8")
    else:
        (results_dir / "STOP_COMMAND.txt").write_text("STOP_FLAGSHIP_PERFORMANCE_CLAIM\nDO_NOT_BEGIN_HARDWARE_OR_GENERAL_TRANSDUCER_WORK\n", encoding="utf-8")

    manifest = []
    for path in sorted(results_dir.iterdir()):
        if path.is_file(): manifest.append(f"{sha256(path)}  {path.name}")
    (results_dir / "RESULTS_SHA256.txt").write_text("\n".join(manifest)+"\n", encoding="utf-8")
    print("="*78)
    print(f"FINAL CLASSIFICATION: {classification}")
    print(f"REASON: {reason}")
    print("="*78)


if __name__ == "__main__":
    main()
