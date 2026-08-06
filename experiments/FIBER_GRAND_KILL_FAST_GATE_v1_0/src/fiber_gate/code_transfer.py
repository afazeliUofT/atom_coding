from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from .baselines import exhaustive_deletion_ml
from .channels import DeletionChannel, sample_deletion_channel
from .codes import SystematicLinearCode, VTCode, crc_systematic_code, hamming_15_11_code
from .history_decoder import deletion_history_model, history_fiber_decode
from .prefix_decoder import prefix_fiber_decode
from .utils import write_json


def _code_suite(rng: np.random.Generator):
    return [
        SystematicLinearCode.random_systematic(16, 12, rng, "TRANSFER_RLC_16_12"),
        crc_systematic_code(16, 12, 0b10011, "TRANSFER_CRC_16_12"),
        hamming_15_11_code(),
        VTCode(15, 0),
    ]


def run_code_transfer(
    output_dir: Path,
    rng: np.random.Generator,
    substitution_probabilities: Sequence[float],
    trials: int,
    max_histories: int,
    max_prefix_nodes: int,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for code in _code_suite(rng):
        for p in substitution_probabilities:
            channel = DeletionChannel(code.n, 1, float(p))
            for trial in range(trials):
                true_message = int(rng.integers(0, code.size))
                transmitted = int(code.codewords_int[true_message])
                received, _, _ = sample_deletion_channel(transmitted, channel, rng)
                reference = exhaustive_deletion_ml(code, received, channel)
                history = history_fiber_decode(
                    deletion_history_model(received, channel),
                    code,
                    reference.tie_set,
                    reference.decision,
                    reference.work,
                    max_histories=max_histories,
                )
                algorithms = {"HISTORY_L0": history}
                if isinstance(code, SystematicLinearCode):
                    algorithms["PREFIX_L2"] = prefix_fiber_decode(
                        received,
                        channel,
                        code,
                        reference.tie_set,
                        reference.decision,
                        reference.work,
                        max_nodes=max_prefix_nodes,
                        use_prefix_feasibility=True,
                    )
                for algorithm, result in algorithms.items():
                    balanced = result.work.scalar("balanced")
                    reference_balanced = reference.work.scalar("balanced")
                    rows.append(
                        {
                            "code": code.name,
                            "code_family": code.family,
                            "n": code.n,
                            "size": code.size,
                            "rate": code.rate,
                            "p": p,
                            "trial": trial,
                            "algorithm": algorithm,
                            "exact": result.exact,
                            "certified": result.certified,
                            "fallback": result.fallback_used,
                            "work_balanced": balanced,
                            "reference_work_balanced": reference_balanced,
                            "wall_seconds": result.work.wall_seconds,
                            "reference_wall_seconds": reference.work.wall_seconds,
                            "wall_speedup": reference.work.wall_seconds / max(result.work.wall_seconds, 1e-12),
                            "speedup": reference_balanced / max(balanced, 1e-12),
                            "candidate_score_speedup": code.size / max(1, result.work.exact_score_calls),
                            "membership_fraction": result.work.membership_queries / code.size,
                            "histories": result.work.history_components,
                            "prefix_nodes": result.work.prefix_nodes,
                        }
                    )
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "07_code_transfer_trials.csv.gz", index=False, compression="gzip")
    summaries: dict[str, Any] = {}
    summary_rows = []
    for (family, p, algorithm), group in frame.groupby(["code_family", "p", "algorithm"]):
        row = {
            "code_family": family,
            "p": float(p),
            "algorithm": algorithm,
            "n": int(group["n"].iloc[0]),
            "size": int(group["size"].iloc[0]),
            "rate": float(group["rate"].iloc[0]),
            "exact_fraction": float(np.mean(group["exact"])),
            "completion_fraction": float(np.mean(~group["fallback"])),
            "mean_speedup": float(np.mean(group["speedup"])),
            "p05_speedup": float(np.percentile(group["speedup"], 5)),
            "median_candidate_score_speedup": float(np.median(group["candidate_score_speedup"])),
            "median_membership_fraction": float(np.median(group["membership_fraction"])),
            "p99_membership_fraction": float(np.percentile(group["membership_fraction"], 99)),
            "mean_wall_speedup": float(np.mean(group["wall_speedup"])),
            "median_wall_speedup": float(np.median(group["wall_speedup"])),
        }
        summary_rows.append(row)
        summaries[f"{family}|p={p}|{algorithm}"] = row
    pd.DataFrame(summary_rows).to_csv(output_dir / "07_code_transfer_summary.csv", index=False)
    payload = {"results": summaries, "exact_all": bool(frame["exact"].all())}
    write_json(output_dir / "07_code_transfer.json", payload)
    return payload
