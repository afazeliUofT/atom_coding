from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .utils import write_json


def make_plots(results_dir: Path) -> None:
    figures = []
    primary = pd.read_csv(results_dir / "02_primary_summary.csv")
    for family in sorted(primary["family"].unique()):
        subset = primary[(primary["family"] == family) & (primary["p"] == 0.05)].sort_values("n")
        if subset.empty:
            continue
        fig = plt.figure(figsize=(6.5, 4.2))
        ax = fig.add_subplot(111)
        ax.semilogy(subset["n"], subset["median_fiber_wall"], marker="o", label="Membership-only FIBER")
        ax.semilogy(subset["n"], subset["median_trellis_wall"], marker="s", label="Syndrome-trellis aggregate")
        ax.semilogy(subset["n"], subset["median_prefix_wall"], marker="^", label="Code-aware prefix A*")
        ax.set_xlabel("Blocklength n")
        ax.set_ylabel("Median Python wall time (s)")
        ax.set_title(f"Strong exact baseline comparison: {family}, p=0.05")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
        fig.tight_layout()
        name = f"FIG_01_primary_{family}.png"
        fig.savefig(results_dir / name, dpi=170)
        plt.close(fig)
        figures.append(name)

    theorem = pd.read_csv(results_dir / "05_shell_theorem_summary.csv")
    fig = plt.figure(figsize=(6.5, 4.2))
    ax = fig.add_subplot(111)
    for (t, p), group in theorem.groupby(["t", "p"]):
        group = group.sort_values("n")
        ax.plot(group["n"], group["p95_excess_over_h2"], marker="o", label=f"t={int(t)}, p={p:.2f}")
    ax.axhline(0.0, linewidth=1)
    ax.set_xlabel("Blocklength n")
    ax.set_ylabel("P95 normalized log-bound minus h2(p)")
    ax.set_title("Polynomial fixed-edit overhead above the BSC exponent")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    name = "FIG_02_shell_theorem.png"
    fig.savefig(results_dir / name, dpi=170)
    plt.close(fig)
    figures.append(name)

    write_json(results_dir / "FIGURES.json", {"figures": figures})
