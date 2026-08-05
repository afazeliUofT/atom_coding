from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _save(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def generate_plots(results_dir: Path) -> list[str]:
    created: list[str] = []

    atlas_path = results_dir / "02_small_dmc_atlas.csv"
    if atlas_path.exists():
        frame = pd.read_csv(atlas_path)
        nonadd = frame[frame["nonadditive"] == True]  # noqa: E712
        if not nonadd.empty:
            fig, ax = plt.subplots(figsize=(7.2, 5.0))
            for family, group in nonadd.groupby("family"):
                ax.scatter(group["kappa"], group["best_work_balanced"], label=family, alpha=0.75)
            ax.set_xlabel("Fiber collision factor kappa")
            ax.set_ylabel("Best expected one-shot balanced work")
            ax.set_title("Exact small-DMC representation atlas")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.25)
            path = results_dir / "FIG_01_atlas_kappa_vs_work.png"
            _save(fig, path)
            created.append(path.name)

    reversible_path = results_dir / "04_reversible_action_summary.csv"
    if reversible_path.exists():
        frame = pd.read_csv(reversible_path)
        if not frame.empty:
            fig, ax = plt.subplots(figsize=(7.2, 5.0))
            for channel, group in frame.groupby("channel"):
                group = group.sort_values("n")
                ax.plot(group["n"], np.log2(group["mean_atom_work_balanced"]), marker="o", label=f"atom: {channel}")
                ax.plot(group["n"], np.log2(group["mean_direct_work_balanced"]), marker="x", linestyle="--", label=f"direct: {channel}")
            ax.set_xlabel("Blocklength n")
            ax.set_ylabel("log2 mean balanced work")
            ax.set_title("Reversible-action positive controls")
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.25)
            path = results_dir / "FIG_02_reversible_scaling.png"
            _save(fig, path)
            created.append(path.name)

    nonlatin_path = results_dir / "05_nonlatin_summary.csv"
    if nonlatin_path.exists():
        frame = pd.read_csv(nonlatin_path)
        if not frame.empty:
            # Select the representation with the smallest largest-n mean work per channel/rate.
            selected_rows = []
            for (channel, target_rate), group in frame.groupby(["channel", "target_rate"]):
                largest_n = group["n"].max()
                final = group[group["n"] == largest_n]
                best_rep = final.loc[final["mean_atom_work_balanced"].idxmin(), "representation"]
                selected_rows.append(group[group["representation"] == best_rep])
            selected = pd.concat(selected_rows, ignore_index=True)
            fig, ax = plt.subplots(figsize=(7.5, 5.2))
            for (channel, target_rate), group in selected.groupby(["channel", "target_rate"]):
                group = group.sort_values("n")
                ax.plot(
                    group["n"],
                    np.log2(group["mean_atom_work_balanced"]),
                    marker="o",
                    label=f"{channel}, R={target_rate:g}",
                )
            ax.set_xlabel("Blocklength n")
            ax.set_ylabel("log2 mean balanced atom work")
            ax.set_title("Best sampled non-Latin representation per channel/rate")
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.25)
            path = results_dir / "FIG_03_nonlatin_scaling.png"
            _save(fig, path)
            created.append(path.name)

            fig, ax = plt.subplots(figsize=(7.5, 5.2))
            for (channel, target_rate), group in selected.groupby(["channel", "target_rate"]):
                group = group.sort_values("n")
                ax.plot(group["n"], group["mean_speedup"], marker="o", label=f"{channel}, R={target_rate:g}")
            ax.axhline(1.0, linestyle="--")
            ax.set_xlabel("Blocklength n")
            ax.set_ylabel("Direct-ML work / atom work")
            ax.set_title("Fully accounted balanced-work speedup")
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.25)
            path = results_dir / "FIG_04_nonlatin_speedup.png"
            _save(fig, path)
            created.append(path.name)

    rate_path = results_dir / "06_rate_survival_analytic.csv"
    if rate_path.exists():
        frame = pd.read_csv(rate_path)
        if not frame.empty:
            best = frame.loc[frame.groupby("channel")["fiber_ceiling_to_capacity"].idxmax()]
            fig, ax = plt.subplots(figsize=(7.2, 5.0))
            x = np.arange(len(best))
            ax.bar(x, best["fiber_ceiling_to_capacity"])
            ax.axhline(0.6, linestyle="--")
            ax.set_xticks(x)
            ax.set_xticklabels(best["channel"], rotation=30, ha="right")
            ax.set_ylabel("Best random-code fiber ceiling / capacity")
            ax.set_title("Analytic rate-survival diagnostic")
            ax.grid(True, axis="y", alpha=0.25)
            path = results_dir / "FIG_05_rate_survival.png"
            _save(fig, path)
            created.append(path.name)

    return created
