from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def _save(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def make_plots(results_dir: Path) -> dict[str, str]:
    figures: dict[str, str] = {}

    rank_path = results_dir / "02_repaired_rank_atlas.csv"
    if rank_path.exists():
        frame = pd.read_csv(rank_path)
        fig, ax = plt.subplots(figsize=(7.2, 4.6))
        ax.plot(frame["n"], frame["worst_normalized_rank_regret"], marker="o", label="worst")
        ax.plot(frame["n"], frame["median_log2_rank_regret"] / frame["n"], marker="s", label="median")
        ax.set_xlabel("Blocklength n")
        ax.set_ylabel("LZ rank regret per symbol")
        ax.set_title("Tie-invariant LZ rank regret versus fitted finite-state comparators")
        ax.grid(True, alpha=0.3)
        ax.legend()
        path = results_dir / "FIG_01_repaired_rank_regret.png"
        _save(fig, path)
        figures["repaired_rank_regret"] = path.name

    mask_path = results_dir / "03_masking_heuristic.csv"
    if mask_path.exists():
        frame = pd.read_csv(mask_path)
        fig, ax = plt.subplots(figsize=(7.2, 4.6))
        for metric, group in frame.groupby("metric"):
            group = group.sort_values("n")
            ax.plot(group["n"], group["max_positive_defect_per_symbol"], marker="o", label=metric)
        ax.set_xscale("log", base=2)
        ax.set_xlabel("Blocklength n")
        ax.set_ylabel("Maximum positive XOR defect / n")
        ax.set_title("Adversarial masking-defect search")
        ax.grid(True, alpha=0.3)
        ax.legend()
        path = results_dir / "FIG_02_masking_defects.png"
        _save(fig, path)
        figures["masking_defects"] = path.name

    stationary_path = results_dir / "04_stationary_rank_summary.csv"
    if stationary_path.exists():
        frame = pd.read_csv(stationary_path)
        subset = frame[(frame["regime"] == "STICKY_BALANCED") & (frame["rate"] == frame["rate"].min())]
        fig, ax = plt.subplots(figsize=(7.2, 4.6))
        for metric in ("ORACLE_MARKOV", "KT1_UNIVERSAL", "KT0_MEMORYLESS", "FIT_16", "FIT_512"):
            group = subset[subset["metric"] == metric].sort_values("n")
            if not group.empty:
                ax.plot(group["n"], group["median_log2_rank"] / group["n"], marker="o", label=metric)
        ax.set_xlabel("Blocklength n")
        ax.set_ylabel("Median log2 level-set rank / n")
        ax.set_title("Universal ranking on sticky Markov impairment")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        path = results_dir / "FIG_03_stationary_rank.png"
        _save(fig, path)
        figures["stationary_rank"] = path.name

    adaptation_path = results_dir / "05_nonstationary_summary.csv"
    if adaptation_path.exists():
        frame = pd.read_csv(adaptation_path).sort_values("mean_log2_rank")
        fig, ax = plt.subplots(figsize=(7.2, 4.6))
        ax.bar(frame["metric"], frame["mean_log2_rank"])
        ax.set_ylabel("Mean log2 level-set rank")
        ax.set_title("Abrupt regime-switching benchmark")
        ax.tick_params(axis="x", rotation=25)
        ax.grid(True, axis="y", alpha=0.3)
        path = results_dir / "FIG_04_nonstationary_adaptation.png"
        _save(fig, path)
        figures["nonstationary_adaptation"] = path.name

    geometry_path = results_dir / "06_code_geometry_errors.csv.gz"
    if geometry_path.exists():
        frame = pd.read_csv(geometry_path)
        subset = frame[(frame["metric"] == "LZ78_FIXED_BLOCK") & (frame["n"] == frame["n"].max())]
        subset = subset[subset["rate"] == subset["rate"].max()]
        grouped = subset.groupby(["code_label", "error_family"])["mdl_unique_correction"].mean().reset_index()
        labels = [f"{row.code_label}\n{row.error_family}" for row in grouped.itertuples()]
        fig, ax = plt.subplots(figsize=(7.2, 4.6))
        ax.bar(labels, grouped["mdl_unique_correction"])
        ax.set_ylim(0, 1)
        ax.set_ylabel("Unique MDL correction fraction")
        ax.set_title("Finite-block description-length code geometry")
        ax.tick_params(axis="x", rotation=20)
        ax.grid(True, axis="y", alpha=0.3)
        path = results_dir / "FIG_05_code_geometry.png"
        _save(fig, path)
        figures["code_geometry"] = path.name

    entropy_path = results_dir / "07_synthetic_entropy_audit.csv"
    if entropy_path.exists():
        frame = pd.read_csv(entropy_path)
        fig, ax = plt.subplots(figsize=(7.2, 4.6))
        x = range(len(frame))
        width = 0.25
        ax.bar([v-width for v in x], frame["marginal_entropy"], width=width, label="marginal H")
        ax.bar(list(x), frame["shannon_entropy_rate"], width=width, label="entropy rate")
        ax.bar([v+width for v in x], frame["renyi_half_rate"], width=width, label="Renyi-1/2 rate")
        ax.set_xticks(list(x), frame["source"], rotation=25)
        ax.set_ylabel("Bits per symbol")
        ax.set_title("Synthetic impairment entropy audit")
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend()
        path = results_dir / "FIG_06_entropy_audit.png"
        _save(fig, path)
        figures["entropy_audit"] = path.name

    (results_dir / "FIGURES.json").write_text(json.dumps(figures, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return figures
