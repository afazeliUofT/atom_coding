from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .utils import write_json


def make_plots(results_dir: Path) -> list[dict[str, str]]:
    figures: list[dict[str, str]] = []

    moment = pd.read_csv(results_dir / "02_moment_rates.csv")
    subset = moment[(moment["t"] == 1) & (moment["rho"] == 1.0)]
    fig, ax = plt.subplots(figsize=(7.4, 4.7))
    for p, group in subset.groupby("p"):
        group = group.sort_values("n")
        ax.plot(group["n"], group["reveal_upper_rate"], marker="o", label=f"reveal p={p:g}")
        ax.plot(group["n"], group["theory_rate"], linestyle="--")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("blocklength n")
    ax.set_ylabel("normalized log2 mean reveal work")
    ax.set_title("Fixed-edit reveal moment convergence")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    path = results_dir / "FIG_01_moment_convergence.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    figures.append({"file": path.name, "description": "Finite-n history-revelation moment rates approaching the Renyi target."})

    phase = pd.read_csv(results_dir / "03_phase_thresholds.csv")
    fig, ax = plt.subplots(figsize=(6.7, 4.4))
    ax.plot(phase["rate"], phase["typical_threshold_p_h2_equals_1_minus_R"], marker="o", label="typical/fixed quantile")
    ax.plot(phase["rate"], phase["mean_threshold_p_Hhalf_equals_1_minus_R"], marker="s", label="mean work")
    ax.set_xlabel("code rate R")
    ax.set_ylabel("substitution crossover p")
    ax.set_title("Typical and mean complexity phase boundaries")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    path = results_dir / "FIG_02_phase_diagram.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    figures.append({"file": path.name, "description": "Rate-dependent typical and mean channel-side/code-side crossover probabilities."})

    summary = pd.read_csv(results_dir / "03_compiled_summary.csv")
    largest = summary[summary["n"] == summary["n"].max()].copy()
    largest["label"] = largest.apply(lambda r: f"{r['family']} R={r['rate']:.3f} p={r['p']:.3f}", axis=1)
    fig, ax = plt.subplots(figsize=(9.4, 5.2))
    x = np.arange(len(largest))
    ax.bar(x, largest["median_fiber_over_best_wall"])
    ax.axhline(1.0, linestyle="--", linewidth=1)
    ax.axhline(0.8, linestyle=":", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(largest["label"], rotation=70, ha="right", fontsize=7)
    ax.set_ylabel("median compiled FIBER / best exact baseline wall time")
    ax.set_title(f"Equally compiled exact-decoder comparison (n={int(largest['n'].max())})")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    path = results_dir / "FIG_03_compiled_benchmark.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    figures.append({"file": path.name, "description": "C++20 FIBER wall-time ratio versus the faster available compiled exact prefix-A* or syndrome-trellis baseline."})

    audit = pd.read_csv(results_dir / "01_search_inflation_audit.csv")
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    if not audit.empty:
        ax.scatter(audit["exact_oracle_lower_volume"], audit["fiber_distinct_candidates"], alpha=0.7)
        maximum = max(float(audit["fiber_distinct_candidates"].max()), float(audit["exact_oracle_lower_volume"].max()), 1.0)
        ax.plot([1, maximum], [1, maximum], linestyle="--", linewidth=1)
        ax.set_xscale("symlog", linthresh=1)
        ax.set_yscale("symlog", linthresh=1)
    ax.set_xlabel("mandatory higher-likelihood candidate queries (exact small world)")
    ax.set_ylabel("FIBER distinct candidate queries")
    ax.set_title("Oracle-relative candidate-query audit")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = results_dir / "FIG_04_oracle_competitiveness.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    figures.append({"file": path.name, "description": "Small-world audit of FIBER candidate queries against the L0 membership-oracle lower bound."})

    write_json(results_dir / "FIGURES.json", figures)
    return figures
