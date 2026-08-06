from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .phase_diagram import threshold_for_target
from .utils import slope_log2, write_json


def _run_checked(command: list[str], cwd: Path, log_path: Path) -> str:
    started = time.perf_counter()
    process = subprocess.run(
        command,
        cwd=str(cwd),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    elapsed = time.perf_counter() - started
    text = f"$ {' '.join(command)}\n{process.stdout}\nELAPSED_SECONDS={elapsed:.6f}\nRETURN_CODE={process.returncode}\n"
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(text)
    if process.returncode != 0:
        raise RuntimeError(f"command failed ({process.returncode}): {' '.join(command)}\n{process.stdout}")
    return process.stdout


def _bootstrap_median_ci(values: np.ndarray, rng: np.random.Generator, replicates: int = 1000) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    median = float(np.median(values))
    if len(values) < 2:
        return median, median, median
    draws = rng.choice(values, size=(replicates, len(values)), replace=True)
    medians = np.median(draws, axis=1)
    return median, float(np.quantile(medians, 0.025)), float(np.quantile(medians, 0.975))


def _schedule_args(profile: dict[str, Any], output_dir: Path, binary: Path, package_root: Path, log_path: Path) -> list[Path]:
    files: list[Path] = []
    for index, row in enumerate(profile["compiled_schedules"]):
        output = output_dir / f"03_compiled_trials_part{index:02d}.csv"
        command = [
            str(binary),
            "--output", str(output),
            "--seed", str(int(profile["seed"]) + index * 100003),
            "--n", ",".join(str(v) for v in row["blocklengths"]),
            "--rates", ",".join(str(v) for v in row["rates"]),
            "--p", ",".join(str(v) for v in row["probabilities"]),
            "--families", ",".join(row["families"]),
            "--trials32", str(int(row.get("trials32", row.get("trials", 10)))),
            "--trials48", str(int(row.get("trials48", row.get("trials", 10)))),
            "--trials64", str(int(row.get("trials64", row.get("trials", 10)))),
            "--timing-repeats", str(int(profile["timing_repeats"])),
            "--max-histories", str(int(row.get("max_histories", profile["max_histories"]))),
            "--max-prefix-nodes", str(int(row.get("max_prefix_nodes", profile["max_prefix_nodes"]))),
            "--max-trellis-terminals", str(int(row.get("max_trellis_terminals", profile["max_trellis_terminals"]))),
            "--max-trellis-dp-updates", str(int(row.get("max_trellis_dp_updates", profile["max_trellis_dp_updates"]))),
        ]
        _run_checked(command, package_root, log_path)
        files.append(output)
    return files


def run_compiled_flagship_gate(
    package_root: Path,
    results_dir: Path,
    profile: dict[str, Any],
    contract: dict[str, Any],
    rng: np.random.Generator,
) -> dict[str, Any]:
    build_dir = package_root / "build"
    build_dir.mkdir(exist_ok=True)
    binary = build_dir / "flagship_benchmark"
    source = package_root / "cpp" / "flagship_benchmark.cpp"
    log_path = results_dir / "03_compiled_build_and_run.log"
    log_path.write_text("", encoding="utf-8")

    compiler = shutil.which(str(profile.get("cxx", "g++")))
    if compiler is None:
        raise RuntimeError("A C++20 compiler (g++) is required for the flagship benchmark")
    compile_command = [
        compiler,
        "-O3",
        "-DNDEBUG",
        "-std=c++20",
        "-march=native",
        "-Wall",
        "-Wextra",
        str(source),
        "-o",
        str(binary),
    ]
    _run_checked(compile_command, package_root, log_path)
    self_test_output = _run_checked([str(binary), "--self-test"], package_root, log_path)
    self_test_pass = "SELF_TEST_PASS" in self_test_output

    parts = _schedule_args(profile, results_dir, binary, package_root, log_path)
    frames = [pd.read_csv(path) for path in parts]
    frame = pd.concat(frames, ignore_index=True)
    frame.to_csv(results_dir / "03_compiled_trials.csv.gz", index=False, compression="gzip")

    summary_rows: list[dict[str, Any]] = []
    for (family, rate, n, p), group in frame.groupby(["family", "rate", "n", "p"]):
        ratio = group["fiber_over_best_wall"].to_numpy(float)
        median, ci_low, ci_high = _bootstrap_median_ci(ratio, rng, int(profile["bootstrap_replicates"]))
        redundancy = 1.0 - float(rate)
        p_typ = threshold_for_target(redundancy, "typical")
        p_mean = threshold_for_target(redundancy, "mean")
        summary_rows.append(
            {
                "family": family,
                "rate": float(rate),
                "n": int(n),
                "p": float(p),
                "trials": int(len(group)),
                "agreement_fraction": float(group["agreement"].mean()),
                "completion_fraction": float((group["fiber_certified"] & (group["prefix_certified"] | group["trellis_certified"])).mean()),
                "median_fiber_over_best_wall": median,
                "median_ratio_ci95_low": ci_low,
                "median_ratio_ci95_high": ci_high,
                "p90_fiber_over_best_wall": float(np.quantile(ratio, 0.90)),
                "p95_fiber_over_best_wall": float(np.quantile(ratio, 0.95)),
                "p99_fiber_over_best_wall": float(np.quantile(ratio, 0.99)),
                "maximum_fiber_over_best_wall": float(np.max(ratio)),
                "median_fiber_wall": float(group["fiber_wall"].median()),
                "median_prefix_wall": float(group["prefix_wall"].median()),
                "median_best_baseline_wall": float(group["best_baseline_wall"].median()),
                "trellis_available_fraction": float(group["trellis_available"].mean()),
                "median_histories": float(group["fiber_histories"].median()),
                "p95_histories": float(group["fiber_histories"].quantile(0.95)),
                "p99_histories": float(group["fiber_histories"].quantile(0.99)),
                "median_membership_queries": float(group["fiber_membership"].median()),
                "median_prefix_nodes": float(group["prefix_nodes"].median()),
                "typical_threshold_p": p_typ,
                "mean_threshold_p": p_mean,
                "predicted_typical_favorable": bool(float(p) < p_typ),
                "predicted_mean_favorable": bool(float(p) < p_mean),
            }
        )
    summary = pd.DataFrame(summary_rows).sort_values(["family", "rate", "p", "n"])
    summary.to_csv(results_dir / "03_compiled_summary.csv", index=False)

    slope_rows: list[dict[str, Any]] = []
    for (family, rate, p), group in summary.groupby(["family", "rate", "p"]):
        group = group.sort_values("n")
        if len(group) < 2:
            continue
        fiber_slope = slope_log2(group["n"], group["median_fiber_wall"])
        prefix_slope = slope_log2(group["n"], group["median_best_baseline_wall"])
        ratio_slope = slope_log2(group["n"], group["median_fiber_over_best_wall"])
        slope_rows.append(
            {
                "family": family,
                "rate": float(rate),
                "p": float(p),
                "fiber_wall_log2_slope": fiber_slope,
                "best_baseline_wall_log2_slope": prefix_slope,
                "fiber_minus_prefix_slope": fiber_slope - prefix_slope,
                "ratio_log2_slope": ratio_slope,
                "predicted_typical_favorable": bool(group["predicted_typical_favorable"].iloc[0]),
                "predicted_mean_favorable": bool(group["predicted_mean_favorable"].iloc[0]),
            }
        )
    slopes = pd.DataFrame(slope_rows)
    slopes.to_csv(results_dir / "03_compiled_slopes.csv", index=False)

    largest_n = int(summary["n"].max())
    largest = summary[summary["n"] == largest_n]
    favorable = largest[largest["predicted_typical_favorable"]]
    mean_favorable = largest[largest["predicted_mean_favorable"]]
    c = contract["compiled_benchmark"]
    median_win_count = int((favorable["median_fiber_over_best_wall"] <= float(c["median_win_ratio"])).sum())
    robust_ci_count = int((favorable["median_ratio_ci95_high"] <= float(c["maximum_median_ci_high"])).sum())
    p95_win_count = int((favorable["p95_fiber_over_best_wall"] <= float(c["maximum_p95_ratio"])).sum())
    slope_pass_count = int(
        (
            slopes[slopes["predicted_typical_favorable"]]["fiber_minus_prefix_slope"]
            <= float(c["maximum_slope_disadvantage"])
        ).sum()
    ) if not slopes.empty else 0
    pass_flag = bool(
        self_test_pass
        and bool(frame["agreement"].all())
        and bool((frame["fiber_certified"] & (frame["prefix_certified"] | frame["trellis_certified"])).all())
        and len(favorable) >= int(c["minimum_favorable_largest_n_configs"])
        and median_win_count >= int(c["minimum_median_wins"])
        and robust_ci_count >= int(c["minimum_robust_ci_wins"])
        and p95_win_count >= int(c["minimum_p95_wins"])
        and slope_pass_count >= int(c["minimum_slope_passes"])
        and (
            mean_favorable.empty
            or float(mean_favorable["median_fiber_over_best_wall"].max())
            <= float(c["maximum_mean_favorable_median_ratio"])
        )
    )

    payload = {
        "pass": pass_flag,
        "compiler": compiler,
        "compile_flags": compile_command[1:-3],
        "self_test_pass": self_test_pass,
        "exact_agreement_all": bool(frame["agreement"].all()),
        "completion_all": bool((frame["fiber_certified"] & (frame["prefix_certified"] | frame["trellis_certified"])).all()),
        "rows": int(len(frame)),
        "largest_n": largest_n,
        "largest_n_favorable_rows": favorable.to_dict(orient="records"),
        "largest_n_mean_favorable_rows": mean_favorable.to_dict(orient="records"),
        "median_win_count": median_win_count,
        "robust_ci_win_count": robust_ci_count,
        "p95_win_count": p95_win_count,
        "slope_pass_count": slope_pass_count,
        "interpretation": (
            "FIBER, prefix A*, and the syndrome-trellis implementation are compiled from the same C++20 translation unit. "
            "Prefix A* is the authoritative timed reference at the largest lengths; syndrome trellis is timed where its DP budget is feasible and is independently exercised in the exhaustive self-test. "
            "Decoder order is randomized. This materially reduces the Python-interpreter confound, but the comparison remains a reference implementation rather than a hardware or universally optimized state-of-the-art result."
        ),
    }
    write_json(results_dir / "03_compiled_gate.json", payload)
    return payload
