from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .utils import write_json


def _save(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def create_figures(results_dir: Path) -> dict[str, Any]:
    generated: list[str] = []

    boundary_path = results_dir / "03_boundary_summary.csv"
    if boundary_path.exists():
        frame = pd.read_csv(boundary_path)
        labels = [f"{row.profile}\np={row.p:g}, n={int(row.n)}" for row in frame.itertuples()]
        values = frame["pathwise_ml_disagreement_fraction"].to_numpy(float)
        fig, ax = plt.subplots(figsize=(10, 4.8))
        ax.bar(np.arange(len(values)), values)
        ax.set_xticks(np.arange(len(values)))
        ax.set_xticklabels(labels, rotation=55, ha="right", fontsize=7)
        ax.set_ylabel("Pathwise first-hit ML disagreement fraction")
        ax.set_title("Best history is not best aggregate candidate")
        ax.grid(axis="y", alpha=0.3)
        path = results_dir / "FIG_01_pathwise_boundary.png"
        _save(fig, path)
        generated.append(path.name)

    one_path = results_dir / "04_one_deletion_summary.csv"
    if one_path.exists():
        frame = pd.read_csv(one_path)
        subset = frame[(frame["p"].round(10) == 0.05) & (frame["code_family"] == "random_linear")]
        fig, ax = plt.subplots(figsize=(7.5, 5.0))
        for algorithm, group in subset.groupby("algorithm"):
            group = group.sort_values("n")
            ax.plot(group["n"], group["mean_work"], marker="o", label=algorithm)
        reference = subset.groupby("n", as_index=False)["mean_reference_work"].first().sort_values("n")
        ax.plot(reference["n"], reference["mean_reference_work"], marker="s", linestyle="--", label="EXHAUSTIVE_ML")
        ax.set_yscale("log")
        ax.set_xlabel("Blocklength n")
        ax.set_ylabel("Mean balanced work (log scale)")
        ax.set_title("One deletion + substitutions: exact-work scaling")
        ax.legend()
        ax.grid(alpha=0.3)
        path = results_dir / "FIG_02_one_deletion_work.png"
        _save(fig, path)
        generated.append(path.name)

        largest_n = int(frame["n"].max())
        speed = frame[(frame["n"] == largest_n) & (frame["p"].isin([0.02, 0.05]))].copy()
        speed["label"] = speed["code_family"] + "\n" + speed["algorithm"] + "\np=" + speed["p"].astype(str)
        fig, ax = plt.subplots(figsize=(10, 5.0))
        ax.bar(np.arange(len(speed)), speed["mean_speedup"])
        ax.set_yscale("log")
        ax.set_xticks(np.arange(len(speed)))
        ax.set_xticklabels(speed["label"], rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Mean exhaustive-work / FIBER-work")
        ax.set_title(f"One-deletion work reduction at n={largest_n}")
        ax.grid(axis="y", alpha=0.3)
        path = results_dir / "FIG_03_one_deletion_speedup.png"
        _save(fig, path)
        generated.append(path.name)

    insertion_path = results_dir / "05_one_insertion_summary.csv"
    if insertion_path.exists():
        frame = pd.read_csv(insertion_path)
        fig, ax = plt.subplots(figsize=(7.5, 5.0))
        for p, group in frame.groupby("p"):
            group = group.sort_values("n")
            ax.plot(group["n"], group["mean_work"], marker="o", label=f"FIBER p={p:g}")
        reference = frame.groupby("n", as_index=False)["mean_reference_work"].first().sort_values("n")
        ax.plot(reference["n"], reference["mean_reference_work"], marker="s", linestyle="--", label="EXHAUSTIVE_ML")
        ax.set_yscale("log")
        ax.set_xlabel("Blocklength n")
        ax.set_ylabel("Mean balanced work (log scale)")
        ax.set_title("One insertion + substitutions")
        ax.legend()
        ax.grid(alpha=0.3)
        path = results_dir / "FIG_04_one_insertion_work.png"
        _save(fig, path)
        generated.append(path.name)

    two_path = results_dir / "06_two_deletion_summary.csv"
    if two_path.exists():
        frame = pd.read_csv(two_path)
        fig, ax = plt.subplots(figsize=(7.5, 5.0))
        for p, group in frame.groupby("p"):
            group = group.sort_values("n")
            ax.plot(group["n"], group["mean_work"], marker="o", label=f"FIBER p={p:g}")
        reference = frame.groupby("n", as_index=False)["mean_reference_work"].first().sort_values("n")
        ax.plot(reference["n"], reference["mean_reference_work"], marker="s", linestyle="--", label="EXHAUSTIVE_ML")
        ax.set_yscale("log")
        ax.set_xlabel("Blocklength n")
        ax.set_ylabel("Mean balanced work (log scale)")
        ax.set_title("Two deletions + substitutions")
        ax.legend()
        ax.grid(alpha=0.3)
        path = results_dir / "FIG_05_two_deletion_work.png"
        _save(fig, path)
        generated.append(path.name)

    transfer_path = results_dir / "07_code_transfer_summary.csv"
    if transfer_path.exists():
        frame = pd.read_csv(transfer_path)
        subset = frame[frame["p"].round(10) == 0.05].copy()
        subset["label"] = subset["code_family"] + "\n" + subset["algorithm"]
        fig, ax = plt.subplots(figsize=(9.5, 5.0))
        ax.bar(np.arange(len(subset)), subset["mean_speedup"])
        ax.set_yscale("log")
        ax.set_xticks(np.arange(len(subset)))
        ax.set_xticklabels(subset["label"], rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Mean exhaustive-work / FIBER-work")
        ax.set_title("Transfer across unmodified code families (p=0.05)")
        ax.grid(axis="y", alpha=0.3)
        path = results_dir / "FIG_06_code_transfer.png"
        _save(fig, path)
        generated.append(path.name)

    payload = {"generated": generated, "count": len(generated)}
    write_json(results_dir / "FIGURES.json", payload)
    return payload
