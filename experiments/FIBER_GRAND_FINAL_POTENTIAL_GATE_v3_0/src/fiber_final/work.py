from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class Work:
    histories: int = 0
    distinct_candidates: int = 0
    duplicates: int = 0
    membership_queries: int = 0
    syndrome_bitops: int = 0
    exact_score_calls: int = 0
    likelihood_ops: int = 0
    candidate_build_ops: int = 0
    heap_pushes: int = 0
    heap_pops: int = 0
    bound_checks: int = 0
    trellis_dp_updates: int = 0
    trellis_nodes: int = 0
    trellis_terminals: int = 0
    vt_candidate_checks: int = 0
    peak_frontier: int = 0
    peak_seen: int = 0
    wall_seconds: float = 0.0
    peak_rss_bytes: int = 0

    def scalar(self) -> float:
        # Frozen, intentionally transparent operation proxy. Wall time is always
        # reported separately and carries greater weight in the final decision.
        return float(
            self.histories
            + 0.5 * self.distinct_candidates
            + 0.1 * self.duplicates
            + 1.5 * self.membership_queries
            + 0.05 * self.syndrome_bitops
            + 0.5 * self.exact_score_calls
            + 0.35 * self.likelihood_ops
            + 0.4 * self.candidate_build_ops
            + 0.35 * self.heap_pushes
            + 0.6 * self.heap_pops
            + 0.2 * self.bound_checks
            + 0.15 * self.trellis_dp_updates
            + 0.5 * self.trellis_nodes
            + 0.5 * self.trellis_terminals
            + 0.5 * self.vt_candidate_checks
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["scalar"] = self.scalar()
        return payload
