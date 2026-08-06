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
    ax.set_title("Fixed-edit reveal moment convergence (rho=1, t=1)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    path = results_dir / "FIG_01_moment_convergence.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    figures.append({"file": path.name, "description": "Exact finite-n reveal moment rates approaching the Renyi target."})

    phase = pd.read_csv(results_dir / "03_phase_thresholds.csv")
    fig, ax = plt.subplots(figsize=(6.7, 4.4))
    ax.plot(phase["rate"], phase["typical_threshold_p_h2_equals_1_minus_R"], marker="o", label="typical / fixed quantile")
    ax.plot(phase["rate"], phase["mean_threshold_p_Hhalf_equals_1_minus_R"], marker="s", label="mean work")
    ax.set_xlabel("code rate R")
    ax.set_ylabel("substitution crossover p")
    ax.set_title("Distinct typical and mean complexity phase boundaries")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    path = results_dir / "FIG_02_phase_diagram.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    figures.append({"file": path.name, "description": "Rate-dependent typical and mean FIBER/code-side crossover probabilities."})

    summary = pd.read_csv(results_dir / "04_primary_summary.csv")
    largest = summary[summary["n"] == summary["n"].max()].copy()
    largest["label"] = largest.apply(lambda r: f"{r['family']} R={r['rate']:.3f} p={r['p']:.3f}", axis=1)
    fig, ax = plt.subplots(figsize=(9.2, 5.1))
    x = np.arange(len(largest))
    ax.bar(x, largest["median_fiber_over_best_wall"])
    ax.axhline(1.0, linestyle="--", linewidth=1)
    ax.axhline(0.8, linestyle=":", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(largest["label"], rotation=70, ha="right", fontsize=7)
    ax.set_ylabel("median FIBER / best exact wall time")
    ax.set_title(f"Largest-block exact-decoder comparison (n={int(largest['n'].max())})")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    path = results_dir / "FIG_03_exact_benchmark.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    figures.append({"file": path.name, "description": "FIBER wall-time ratio versus the faster of syndrome-trellis and prefix A*."})

    physical = pd.read_csv(results_dir / "07_physical_position_calibration.csv")
    fig, ax = plt.subplots(figsize=(7.4, 4.5))
    ax.plot(physical["position"], physical["q_train"], label="training fit")
    ax.plot(physical["position"], physical["q_test"], label="held out")
    ax.plot(physical["position"], physical["uniform"], linestyle="--", label="uniform")
    ax.set_xlabel("missed-decision position")
    ax.set_ylabel("probability")
    ax.set_title("Synthetic timing-slip position calibration")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    path = results_dir / "FIG_04_physical_calibration.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    figures.append({"file": path.name, "description": "Training and held-out deletion-position distributions for the synthetic front end."})

    write_json(results_dir / "FIGURES.json", figures)
    return figures
