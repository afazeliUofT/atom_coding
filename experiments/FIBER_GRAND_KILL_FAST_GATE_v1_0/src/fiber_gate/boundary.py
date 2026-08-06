from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from .baselines import exhaustive_deletion_ml
from .channels import DeletionChannel, edge_biased_weights, sample_deletion_channel
from .codes import SystematicLinearCode
from .history_decoder import deletion_history_model, history_fiber_decode, pathwise_first_hit
from .likelihood import deletion_likelihood_vectorized
from .utils import int_array, write_json


def _ideal_rank_metrics(
    code: SystematicLinearCode,
    received: int,
    channel: DeletionChannel,
) -> dict[str, float]:
    ambient_words = np.arange(1 << channel.n, dtype=np.uint64)
    ambient_array = int_array(ambient_words, channel.n)
    ambient_scores, _ = deletion_likelihood_vectorized(ambient_array, received, channel)
    code_scores = ambient_scores[code.codewords_int.astype(np.int64)]
    ml_score = float(np.max(code_scores))
    optimistic = 1 + int(np.count_nonzero(ambient_scores > ml_score + 1e-13))
    pessimistic = int(np.count_nonzero(ambient_scores >= ml_score - 1e-13))
    return {
        "ideal_rank_optimistic": float(optimistic),
        "ideal_rank_pessimistic": float(max(optimistic, pessimistic)),
        "ml_score": ml_score,
    }


def run_boundary_audit(
    output_dir: Path,
    rng: np.random.Generator,
    blocklengths: Sequence[int],
    substitution_probabilities: Sequence[float],
    trials: int,
    max_histories: int,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for n in blocklengths:
        k = max(2, int(round(0.75 * n)))
        code = SystematicLinearCode.random_systematic(n, k, rng, f"BOUNDARY_RLC_{n}_{k}")
        for profile_name, weights in (
            ("uniform", None),
            ("edge_biased", edge_biased_weights(n)),
        ):
            for p in substitution_probabilities:
                channel = DeletionChannel(
                    n,
                    1,
                    float(p),
                    deletion_weights=weights,
                    name=f"D1_{profile_name}_p{p:.3f}_n{n}",
                )
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
                    path_decision, path_work = pathwise_first_hit(
                        deletion_history_model(received, channel),
                        code,
                        max_histories=max_histories,
                    )
                    rank = _ideal_rank_metrics(code, received, channel)
                    rows.append(
                        {
                            "n": n,
                            "k": k,
                            "code_size": code.size,
                            "profile": profile_name,
                            "p": p,
                            "trial": trial,
                            "history_exact": history.exact,
                            "history_certified": history.certified,
                            "history_distinct_candidates": history.work.distinct_candidates,
                            "history_components": history.work.history_components,
                            "history_inflation_optimistic": history.work.distinct_candidates / rank["ideal_rank_optimistic"],
                            "history_inflation_pessimistic": history.work.distinct_candidates / rank["ideal_rank_pessimistic"],
                            "history_per_candidate": history.work.history_components / max(1, history.work.distinct_candidates),
                            "pathwise_exact_ml": path_decision in reference.tie_set if path_decision is not None else False,
                            "pathwise_histories": path_work.history_components,
                            **rank,
                        }
                    )
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "03_boundary_trials.csv.gz", index=False, compression="gzip")
    summaries: dict[str, Any] = {}
    summary_rows = []
    for (profile, p, n), group in frame.groupby(["profile", "p", "n"]):
        row = {
            "profile": profile,
            "p": float(p),
            "n": int(n),
            "trials": len(group),
            "pathwise_ml_disagreement_fraction": float(np.mean(~group["pathwise_exact_ml"])),
            "history_exact_fraction": float(np.mean(group["history_exact"])),
            "history_completion_fraction": float(np.mean(group["history_certified"])),
            "median_search_inflation_optimistic": float(np.median(group["history_inflation_optimistic"])),
            "p95_search_inflation_optimistic": float(np.percentile(group["history_inflation_optimistic"], 95)),
            "median_search_inflation_pessimistic": float(np.median(group["history_inflation_pessimistic"])),
            "median_history_per_candidate": float(np.median(group["history_per_candidate"])),
            "p95_history_per_candidate": float(np.percentile(group["history_per_candidate"], 95)),
        }
        summary_rows.append(row)
        summaries[f"{profile}|p={p}|n={n}"] = row
    pd.DataFrame(summary_rows).to_csv(output_dir / "03_boundary_summary.csv", index=False)
    payload = {"results": summaries, "exact_all": bool(frame["history_exact"].all())}
    write_json(output_dir / "03_boundary_audit.json", payload)
    return payload
