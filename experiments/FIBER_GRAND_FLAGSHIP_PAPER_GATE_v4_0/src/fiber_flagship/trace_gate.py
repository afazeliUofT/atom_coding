from __future__ import annotations

import glob
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .utils import write_json

REQUIRED_COLUMNS = {
    "frame_id",
    "blocklength",
    "edit_count",
    "substitution_count",
}


def run_measured_trace_gate(package_root: Path, results_dir: Path, trace_glob: str) -> dict[str, Any]:
    pattern = str(package_root / trace_glob)
    files = [Path(path) for path in sorted(glob.glob(pattern)) if Path(path).is_file() and Path(path).name != "trace_template.csv"]
    rows: list[pd.DataFrame] = []
    errors: list[str] = []
    for path in files:
        try:
            frame = pd.read_csv(path)
        except Exception as exc:  # pragma: no cover - defensive input handling
            errors.append(f"{path.name}: could not read CSV: {exc}")
            continue
        missing = REQUIRED_COLUMNS.difference(frame.columns)
        if missing:
            errors.append(f"{path.name}: missing columns {sorted(missing)}")
            continue
        frame = frame.copy()
        frame["source_file"] = path.name
        rows.append(frame)

    if not rows:
        payload = {
            "status": "PENDING_MEASURED_REAL_TRACE",
            "files": [path.name for path in files],
            "valid_rows": 0,
            "errors": errors,
            "paper_consequence": (
                "A theory/algorithm flagship submission may proceed without this gate, but no real-system, telecommunications, "
                "or field-defining-impact claim is authorized."
            ),
        }
        write_json(results_dir / "04_measured_trace_gate.json", payload)
        return payload

    frame = pd.concat(rows, ignore_index=True)
    for column in ("blocklength", "edit_count", "substitution_count"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["blocklength", "edit_count", "substitution_count"])
    frame = frame[(frame["blocklength"] > 0) & (frame["edit_count"] >= 0) & (frame["substitution_count"] >= 0)]
    if frame.empty:
        payload = {
            "status": "PENDING_MEASURED_REAL_TRACE",
            "files": [path.name for path in files],
            "valid_rows": 0,
            "errors": errors + ["No valid trace rows remained after numeric validation."],
            "paper_consequence": (
                "A theory/algorithm flagship submission may proceed without this gate, but no real-system, telecommunications, "
                "or field-defining-impact claim is authorized."
            ),
        }
        write_json(results_dir / "04_measured_trace_gate.json", payload)
        return payload
    frame["survivor_length"] = frame["blocklength"] - frame["edit_count"]
    frame = frame[frame["survivor_length"] > 0]
    frame["substitution_fraction"] = frame["substitution_count"] / frame["survivor_length"]
    frame.to_csv(results_dir / "04_measured_trace_rows.csv.gz", index=False, compression="gzip")

    exactly_one = float((frame["edit_count"] == 1).mean()) if len(frame) else 0.0
    bounded_two = float((frame["edit_count"] <= 2).mean()) if len(frame) else 0.0
    p_hat = float(frame["substitution_count"].sum() / frame["survivor_length"].sum()) if len(frame) else float("nan")
    p95 = float(frame["substitution_fraction"].quantile(0.95)) if len(frame) else float("nan")
    payload = {
        "status": "MEASURED_TRACE_PRESENT",
        "files": sorted(frame["source_file"].unique().tolist()),
        "valid_rows": int(len(frame)),
        "exactly_one_edit_fraction": exactly_one,
        "at_most_two_edits_fraction": bounded_two,
        "pooled_substitution_probability": p_hat,
        "p95_frame_substitution_fraction": p95,
        "errors": errors,
        "interpretation": (
            "This audit only establishes whether a measured post-front-end trace occupies the bounded-edit, low-substitution region. "
            "It does not by itself prove an end-to-end decoder advantage or channel-model goodness of fit."
        ),
    }
    write_json(results_dir / "04_measured_trace_gate.json", payload)
    return payload
