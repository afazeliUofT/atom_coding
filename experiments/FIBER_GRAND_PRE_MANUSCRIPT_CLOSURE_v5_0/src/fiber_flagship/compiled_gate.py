from __future__ import annotations

import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import binom

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
        env={
            **os.environ,
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        },
    )
    elapsed = time.perf_counter() - started
    text = (
        f"$ {' '.join(command)}\n{process.stdout}\n"
        f"ELAPSED_SECONDS={elapsed:.6f}\nRETURN_CODE={process.returncode}\n"
    )
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(text)
    if process.returncode != 0:
        raise RuntimeError(
            f"command failed ({process.returncode}): {' '.join(command)}\n{process.stdout}"
        )
    return process.stdout


def _bootstrap_median_ci(
    values: np.ndarray,
    rng: np.random.Generator,
    replicates: int = 2000,
) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    median = float(np.median(values))
    if len(values) < 2:
        return median, median, median
    draws = rng.choice(values, size=(replicates, len(values)), replace=True)
    medians = np.median(draws, axis=1)
    return median, float(np.quantile(medians, 0.025)), float(np.quantile(medians, 0.975))


def _distribution_free_quantile_ci(
    values: np.ndarray,
    quantile: float,
    confidence: float = 0.95,
) -> tuple[float, float, float]:
    """Conservative order-statistic interval for a population quantile."""
    values = np.sort(np.asarray(values, dtype=float))
    n = len(values)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    estimate = float(np.quantile(values, quantile))
    alpha = 1.0 - confidence
    lower_count = int(binom.ppf(alpha / 2.0, n, quantile))
    upper_count = int(binom.ppf(1.0 - alpha / 2.0, n, quantile))
    lower_index = max(0, min(n - 1, lower_count - 1))
    upper_index = max(0, min(n - 1, upper_count))
    return estimate, float(values[lower_index]), float(values[upper_index])


def _schedule_args(
    profile: dict[str, Any],
    output_dir: Path,
    binary: Path,
    package_root: Path,
    log_path: Path,
) -> list[Path]:
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
            "--timing-repeats", str(int(row.get("timing_repeats", profile["timing_repeats"]))),
            "--warmup-trials", str(int(row.get("warmup_trials", profile["warmup_trials"]))),
            "--max-histories", str(int(row.get("max_histories", profile["max_histories"]))),
            "--max-prefix-nodes", str(int(row.get("max_prefix_nodes", profile["max_prefix_nodes"]))),
            "--max-trellis-terminals", str(int(row.get("max_trellis_terminals", profile["max_trellis_terminals"]))),
            "--max-trellis-dp-updates", str(int(row.get("max_trellis_dp_updates", profile["max_trellis_dp_updates"]))),
        ]
        _run_checked(command, package_root, log_path)
        files.append(output)
    return files


def _error_bucket(weight: int) -> str:
    if weight <= 0:
        return "E0"
    if weight == 1:
        return "E1"
    if weight == 2:
        return "E2"
    return "E3PLUS"


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
        raise RuntimeError("A C++20 compiler (g++) is required for the pre-manuscript benchmark")
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
    frame["error_bucket"] = frame["error_weight"].map(_error_bucket)
    frame.to_csv(results_dir / "03_compiled_trials.csv.gz", index=False, compression="gzip")

    summary_rows: list[dict[str, Any]] = []
    for (family, rate, n, p), group in frame.groupby(["family", "rate", "n", "p"]):
        ratio = group["fiber_over_best_wall"].to_numpy(float)
        median, ci_low, ci_high = _bootstrap_median_ci(
            ratio, rng, int(profile["bootstrap_replicates"])
        )
        p99, p99_low, p99_high = _distribution_free_quantile_ci(ratio, 0.99)
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
                "positive_error_trials": int((group["error_weight"] >= 1).sum()),
                "agreement_fraction": float(group["agreement"].mean()),
                "completion_fraction": float(
                    (
                        group["fiber_certified"]
                        & (group["prefix_certified"] | group["trellis_certified"])
                    ).mean()
                ),
                "median_fiber_over_best_wall": median,
                "median_ratio_ci95_low": ci_low,
                "median_ratio_ci95_high": ci_high,
                "p90_fiber_over_best_wall": float(np.quantile(ratio, 0.90)),
                "p95_fiber_over_best_wall": float(np.quantile(ratio, 0.95)),
                "p99_fiber_over_best_wall": p99,
                "p99_ratio_ci95_low": p99_low,
                "p99_ratio_ci95_high": p99_high,
                "maximum_fiber_over_best_wall": float(np.max(ratio)),
                "median_fiber_wall": float(group["fiber_wall"].median()),
                "median_prefix_wall": float(group["prefix_wall"].median()),
                "median_trellis_wall": float(
                    group.loc[group["trellis_certified"] == 1, "trellis_wall"].median()
                    if (group["trellis_certified"] == 1).any()
                    else np.nan
                ),
                "median_best_baseline_wall": float(group["best_baseline_wall"].median()),
                "prefix_best_fraction": float((group["best_baseline_name"] == "PREFIX_ASTAR").mean()),
                "trellis_best_fraction": float((group["best_baseline_name"] == "SYNDROME_TRELLIS").mean()),
                "trellis_available_fraction": float(group["trellis_available"].mean()),
                "trellis_certified_fraction": float(group["trellis_certified"].mean()),
                "trellis_estimated_updates": int(group["trellis_estimated_updates"].iloc[0]),
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

    error_rows: list[dict[str, Any]] = []
    for (family, rate, n, p, bucket), group in frame.groupby(
        ["family", "rate", "n", "p", "error_bucket"]
    ):
        ratio = group["fiber_over_best_wall"].to_numpy(float)
        error_rows.append(
            {
                "family": family,
                "rate": float(rate),
                "n": int(n),
                "p": float(p),
                "error_bucket": bucket,
                "trials": int(len(group)),
                "median_fiber_over_best_wall": float(np.median(ratio)),
                "p90_fiber_over_best_wall": float(np.quantile(ratio, 0.90)),
                "p95_fiber_over_best_wall": float(np.quantile(ratio, 0.95)),
                "p99_fiber_over_best_wall": float(np.quantile(ratio, 0.99)),
                "maximum_fiber_over_best_wall": float(np.max(ratio)),
                "median_histories": float(group["fiber_histories"].median()),
                "p95_histories": float(group["fiber_histories"].quantile(0.95)),
                "trellis_available_fraction": float(group["trellis_available"].mean()),
                "trellis_best_fraction": float(
                    (group["best_baseline_name"] == "SYNDROME_TRELLIS").mean()
                ),
            }
        )
    error_summary = pd.DataFrame(error_rows).sort_values(
        ["family", "rate", "p", "n", "error_bucket"]
    )
    error_summary.to_csv(results_dir / "03_error_weight_summary.csv", index=False)

    slope_rows: list[dict[str, Any]] = []
    for (family, rate, p), group in summary.groupby(["family", "rate", "p"]):
        group = group.sort_values("n")
        if len(group) < 2:
            continue
        fiber_slope = slope_log2(group["n"], group["median_fiber_wall"])
        baseline_slope = slope_log2(group["n"], group["median_best_baseline_wall"])
        ratio_slope = slope_log2(group["n"], group["median_fiber_over_best_wall"])
        availability = group["trellis_available_fraction"].to_numpy(float)
        reference_policy_consistent = bool(
            np.all(availability >= 1.0 - 1e-12) or np.all(availability <= 1e-12)
        )
        slope_rows.append(
            {
                "family": family,
                "rate": float(rate),
                "p": float(p),
                "fiber_wall_log2_slope": fiber_slope,
                "best_baseline_wall_log2_slope": baseline_slope,
                "fiber_minus_best_slope": fiber_slope - baseline_slope,
                "ratio_log2_slope": ratio_slope,
                "trellis_available_all_lengths": bool(np.all(availability >= 1.0 - 1e-12)),
                "reference_policy_consistent": reference_policy_consistent,
                "predicted_typical_favorable": bool(group["predicted_typical_favorable"].iloc[0]),
                "predicted_mean_favorable": bool(group["predicted_mean_favorable"].iloc[0]),
            }
        )
    slopes = pd.DataFrame(slope_rows)
    slopes.to_csv(results_dir / "03_compiled_slopes.csv", index=False)

    c = contract["compiled_benchmark"]
    largest_n = int(summary["n"].max())
    largest = summary[summary["n"] == largest_n]
    key = largest[
        largest["predicted_typical_favorable"]
        & (largest["p"] <= float(c["maximum_key_probability"]))
    ]
    mean_favorable = key[key["predicted_mean_favorable"]]

    positive = error_summary[
        (error_summary["n"] == largest_n)
        & (error_summary["error_bucket"] != "E0")
        & (error_summary["p"] <= float(c["maximum_key_probability"]))
        & (error_summary["trials"] >= int(c["minimum_positive_error_stratum_trials"]))
    ]

    required_trellis_rows: list[dict[str, Any]] = []
    trellis_requirement_status: list[dict[str, Any]] = []
    trellis_coverage_pass = True
    for requirement in c["required_trellis_points"]:
        subset = summary[
            (summary["n"] == int(requirement["n"]))
            & (np.isclose(summary["rate"], float(requirement["rate"])))
            & (
                summary["p"]
                <= float(requirement.get("maximum_p", c["maximum_key_probability"]))
            )
        ]
        rows_for_requirement = subset.to_dict(orient="records")
        required_trellis_rows.extend(rows_for_requirement)
        requirement_pass = bool(
            rows_for_requirement
            and all(
                float(row["trellis_available_fraction"]) >= 1.0
                and float(row["trellis_certified_fraction"]) >= 1.0
                for row in rows_for_requirement
            )
        )
        trellis_coverage_pass = trellis_coverage_pass and requirement_pass
        trellis_requirement_status.append(
            {
                **requirement,
                "matched_rows": len(rows_for_requirement),
                "pass": requirement_pass,
            }
        )

    median_win_count = int(
        (key["median_fiber_over_best_wall"] <= float(c["median_win_ratio"])).sum()
    )
    robust_ci_count = int(
        (key["median_ratio_ci95_high"] <= float(c["maximum_median_ci_high"])).sum()
    )
    p99_win_count = int(
        (key["p99_fiber_over_best_wall"] <= float(c["maximum_p99_ratio"])).sum()
    )
    p99_ci_win_count = int(
        (key["p99_ratio_ci95_high"] <= float(c["maximum_p99_ci_high"])).sum()
    )
    mean_favorable_p99_pass = bool(
        not mean_favorable.empty
        and float(mean_favorable["p99_fiber_over_best_wall"].max())
        <= float(c["maximum_mean_favorable_p99_ratio"])
    )
    slope_pass_count = (
        int(
            (
                slopes[
                    slopes["predicted_typical_favorable"]
                    & slopes["reference_policy_consistent"]
                ]["fiber_minus_best_slope"]
                <= float(c["maximum_slope_disadvantage"])
            ).sum()
        )
        if not slopes.empty
        else 0
    )
    positive_error_pass_count = int(
        (
            (positive["median_fiber_over_best_wall"] <= float(c["positive_error_median_ratio"]))
            & (positive["p95_fiber_over_best_wall"] <= float(c["positive_error_p95_ratio"]))
        ).sum()
    )

    minimum_key_trials_pass = bool(
        not key.empty
        and int(key["trials"].min()) >= int(c["minimum_key_trials_per_config"])
    )
    positive_error_evidence_pass = bool(
        positive_error_pass_count >= int(c["minimum_positive_error_strata_passes"])
    )

    pass_flag = bool(
        self_test_pass
        and bool(frame["agreement"].all())
        and bool(
            (
                frame["fiber_certified"]
                & (frame["prefix_certified"] | frame["trellis_certified"])
            ).all()
        )
        and trellis_coverage_pass
        and minimum_key_trials_pass
        and len(key) >= int(c["minimum_key_largest_n_configs"])
        and median_win_count >= int(c["minimum_median_wins"])
        and robust_ci_count >= int(c["minimum_robust_ci_wins"])
        and p99_win_count >= int(c["minimum_p99_wins"])
        and p99_ci_win_count >= int(c["minimum_p99_ci_wins"])
        and mean_favorable_p99_pass
        and slope_pass_count >= int(c["minimum_slope_passes"])
        and positive_error_evidence_pass
    )

    payload = {
        "pass": pass_flag,
        "compiler": compiler,
        "compile_flags": compile_command[1:-3],
        "self_test_pass": self_test_pass,
        "exact_agreement_all": bool(frame["agreement"].all()),
        "completion_all": bool(
            (
                frame["fiber_certified"]
                & (frame["prefix_certified"] | frame["trellis_certified"])
            ).all()
        ),
        "rows": int(len(frame)),
        "largest_n": largest_n,
        "key_largest_n_rows": key.to_dict(orient="records"),
        "mean_favorable_largest_n_rows": mean_favorable.to_dict(orient="records"),
        "positive_error_rows": positive.to_dict(orient="records"),
        "required_trellis_rows": required_trellis_rows,
        "trellis_requirement_status": trellis_requirement_status,
        "trellis_coverage_pass": trellis_coverage_pass,
        "minimum_key_trials_pass": minimum_key_trials_pass,
        "median_win_count": median_win_count,
        "robust_ci_win_count": robust_ci_count,
        "p99_win_count": p99_win_count,
        "p99_ci_win_count": p99_ci_win_count,
        "mean_favorable_p99_pass": mean_favorable_p99_pass,
        "slope_pass_count": slope_pass_count,
        "positive_error_pass_count": positive_error_pass_count,
        "positive_error_evidence_pass": positive_error_evidence_pass,
        "interpretation": (
            "This closure gate enables the syndrome-trellis reference wherever its charged update count is feasible, "
            "uses per-configuration warmup and randomized decoder order, requires at least 500 key trials, and reports "
            "error-weight-conditioned results so a zero-substitution majority cannot create a false flagship signal. "
            "The comparison remains a reference implementation and does not substitute for independent external code review."
        ),
    }
    write_json(results_dir / "03_compiled_gate.json", payload)
    return payload
