from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class WorkVector:
    history_components: int = 0
    distinct_candidates: int = 0
    duplicate_histories: int = 0
    membership_queries: int = 0
    exact_score_calls: int = 0
    likelihood_symbol_ops: int = 0
    candidate_build_ops: int = 0
    generator_pushes: int = 0
    generator_pops: int = 0
    bound_checks: int = 0
    prefix_nodes: int = 0
    prefix_bound_terms: int = 0
    prefix_feasibility_ops: int = 0
    terminal_candidates: int = 0
    direct_likelihood_ops: int = 0
    direct_candidate_scores: int = 0
    sort_comparisons_proxy: int = 0
    fallback_count: int = 0
    peak_frontier: int = 0
    peak_seen_candidates: int = 0
    peak_rss_bytes: int = 0
    wall_seconds: float = 0.0

    def scalar(self, model: str = "balanced") -> float:
        weights = {
            "optimistic": {
                "history_components": 0.5,
                "distinct_candidates": 0.2,
                "duplicate_histories": 0.05,
                "membership_queries": 1.0,
                "exact_score_calls": 0.2,
                "likelihood_symbol_ops": 0.15,
                "candidate_build_ops": 0.2,
                "generator_pushes": 0.15,
                "generator_pops": 0.25,
                "bound_checks": 0.1,
                "prefix_nodes": 0.3,
                "prefix_bound_terms": 0.1,
                "prefix_feasibility_ops": 0.1,
                "terminal_candidates": 0.2,
                "direct_likelihood_ops": 0.15,
                "direct_candidate_scores": 0.2,
                "sort_comparisons_proxy": 0.05,
                "fallback_count": 0.0,
            },
            "balanced": {
                "history_components": 1.0,
                "distinct_candidates": 0.5,
                "duplicate_histories": 0.1,
                "membership_queries": 1.5,
                "exact_score_calls": 0.5,
                "likelihood_symbol_ops": 0.35,
                "candidate_build_ops": 0.4,
                "generator_pushes": 0.35,
                "generator_pops": 0.6,
                "bound_checks": 0.2,
                "prefix_nodes": 0.6,
                "prefix_bound_terms": 0.25,
                "prefix_feasibility_ops": 0.2,
                "terminal_candidates": 0.5,
                "direct_likelihood_ops": 0.35,
                "direct_candidate_scores": 0.5,
                "sort_comparisons_proxy": 0.1,
                "fallback_count": 0.0,
            },
            "pessimistic": {
                "history_components": 2.0,
                "distinct_candidates": 1.0,
                "duplicate_histories": 0.2,
                "membership_queries": 3.0,
                "exact_score_calls": 1.0,
                "likelihood_symbol_ops": 0.8,
                "candidate_build_ops": 1.0,
                "generator_pushes": 0.8,
                "generator_pops": 1.2,
                "bound_checks": 0.5,
                "prefix_nodes": 1.2,
                "prefix_bound_terms": 0.6,
                "prefix_feasibility_ops": 0.5,
                "terminal_candidates": 1.0,
                "direct_likelihood_ops": 0.8,
                "direct_candidate_scores": 1.0,
                "sort_comparisons_proxy": 0.25,
                "fallback_count": 0.0,
            },
        }
        if model not in weights:
            raise ValueError(f"Unknown work model {model}")
        raw = asdict(self)
        return float(sum(float(raw[key]) * weight for key, weight in weights[model].items()))

    def add(self, other: "WorkVector") -> None:
        for key, value in asdict(other).items():
            if key in ("peak_frontier", "peak_seen_candidates", "peak_rss_bytes"):
                setattr(self, key, max(int(getattr(self, key)), int(value)))
            else:
                setattr(self, key, getattr(self, key) + value)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for model in ("optimistic", "balanced", "pessimistic"):
            payload[f"scalar_{model}"] = self.scalar(model)
        return payload


@dataclass
class DecodeResult:
    decision: int | None
    ml_tie_set: tuple[int, ...]
    exact: bool
    certified: bool
    fallback_used: bool
    incumbent_score: float
    residual_bound: float
    work: WorkVector
    notes: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision": None if self.decision is None else int(self.decision),
            "ml_tie_set": [int(v) for v in self.ml_tie_set],
            "exact": bool(self.exact),
            "certified": bool(self.certified),
            "fallback_used": bool(self.fallback_used),
            "incumbent_score": float(self.incumbent_score),
            "residual_bound": float(self.residual_bound),
            "work": self.work.to_dict(),
            "notes": self.notes,
        }
