from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class DeterministicMap:
    """A finite deterministic channel map represented by output labels per input."""

    outputs: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.outputs:
            raise ValueError("A deterministic map must have at least one input.")
        if min(self.outputs) < 0:
            raise ValueError("Output labels must be nonnegative integers.")

    @property
    def input_size(self) -> int:
        return len(self.outputs)

    @property
    def output_size(self) -> int:
        return max(self.outputs) + 1

    def __call__(self, x: int) -> int:
        return self.outputs[x]

    def preimage(self, y: int) -> tuple[int, ...]:
        return tuple(x for x, out in enumerate(self.outputs) if out == y)

    def is_injective(self, code_symbols: Sequence[int] | None = None) -> bool:
        symbols = tuple(range(self.input_size)) if code_symbols is None else tuple(code_symbols)
        values = [self.outputs[x] for x in symbols]
        return len(values) == len(set(values))


@dataclass
class Representation:
    name: str
    maps: list[DeterministicMap]
    weights: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.weights = np.asarray(self.weights, dtype=float)
        if len(self.maps) != len(self.weights):
            raise ValueError("maps and weights must have the same length")
        if len(self.maps) == 0:
            raise ValueError("A representation must contain at least one atom")
        if np.any(self.weights < -1e-12):
            raise ValueError("Atom weights must be nonnegative")
        if not np.isclose(float(self.weights.sum()), 1.0, atol=1e-9):
            raise ValueError(f"Atom weights sum to {self.weights.sum()}, not 1")
        m = self.maps[0].input_size
        if any(atom.input_size != m for atom in self.maps):
            raise ValueError("All atoms must have the same input alphabet")

    @property
    def input_size(self) -> int:
        return self.maps[0].input_size

    @property
    def output_size(self) -> int:
        return max(atom.output_size for atom in self.maps)

    @property
    def support_size(self) -> int:
        return len(self.maps)

    def reduced(self, atol: float = 1e-15) -> "Representation":
        merged: dict[tuple[int, ...], float] = {}
        for atom, weight in zip(self.maps, self.weights, strict=True):
            if weight <= atol:
                continue
            merged[atom.outputs] = merged.get(atom.outputs, 0.0) + float(weight)
        items = sorted(merged.items())
        weights = np.array([w for _, w in items], dtype=float)
        weights /= weights.sum()
        return Representation(
            name=f"{self.name}__reduced",
            maps=[DeterministicMap(outputs) for outputs, _ in items],
            weights=weights,
            metadata={**self.metadata, "reduced_from": self.name},
        )

    def induced_channel(self, output_size: int | None = None) -> np.ndarray:
        qy = self.output_size if output_size is None else output_size
        channel = np.zeros((self.input_size, qy), dtype=float)
        for atom, weight in zip(self.maps, self.weights, strict=True):
            for x, y in enumerate(atom.outputs):
                if y >= qy:
                    raise ValueError("output_size is too small for an atom")
                channel[x, y] += float(weight)
        return channel

    def verify(self, channel: np.ndarray, atol: float = 2e-10) -> None:
        actual = self.induced_channel(channel.shape[1])
        if not np.allclose(actual, channel, atol=atol, rtol=0.0):
            delta = float(np.max(np.abs(actual - channel)))
            raise AssertionError(
                f"Representation {self.name} does not match channel; max error={delta:.3e}"
            )

    def sorted_by_weight(self) -> "Representation":
        order = sorted(
            range(len(self.maps)),
            key=lambda i: (-float(self.weights[i]), self.maps[i].outputs),
        )
        return Representation(
            name=f"{self.name}__mass_order",
            maps=[self.maps[i] for i in order],
            weights=self.weights[order].copy(),
            metadata={**self.metadata, "order": "mass_descending"},
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "maps": [list(atom.outputs) for atom in self.maps],
            "weights": [float(w) for w in self.weights],
            "metadata": self.metadata,
        }


@dataclass
class WorkVector:
    atoms_processed: int = 0
    generator_pushes: int = 0
    generator_pops: int = 0
    fiber_queries: int = 0
    fiber_solver_bitops: int = 0
    fiber_entries: int = 0
    score_updates: int = 0
    heap_pushes: int = 0
    heap_pops: int = 0
    membership_queries: int = 0
    inverse_symbol_ops: int = 0
    direct_likelihood_ops: int = 0
    trellis_state_updates: int = 0
    trellis_traceback_ops: int = 0
    fallback_count: int = 0
    representation_atoms: int = 0
    wall_seconds: float = 0.0
    peak_rss_bytes: int = 0

    def add(self, other: "WorkVector") -> None:
        for key, value in asdict(other).items():
            setattr(self, key, getattr(self, key) + value)

    def scalar(self, model: str = "balanced") -> float:
        weights = {
            "optimistic": {
                "atoms_processed": 1.0,
                "generator_pushes": 0.2,
                "generator_pops": 0.5,
                "fiber_queries": 1.0,
                "fiber_solver_bitops": 0.02,
                "fiber_entries": 1.0,
                "score_updates": 0.5,
                "heap_pushes": 0.2,
                "heap_pops": 0.2,
                "membership_queries": 1.0,
                "inverse_symbol_ops": 0.1,
                "direct_likelihood_ops": 1.0,
                "trellis_state_updates": 0.5,
                "trellis_traceback_ops": 0.2,
                "fallback_count": 0.0,
                "representation_atoms": 0.0,
            },
            "balanced": {
                "atoms_processed": 1.0,
                "generator_pushes": 0.5,
                "generator_pops": 1.0,
                "fiber_queries": 2.0,
                "fiber_solver_bitops": 0.05,
                "fiber_entries": 2.0,
                "score_updates": 1.0,
                "heap_pushes": 0.5,
                "heap_pops": 0.5,
                "membership_queries": 1.5,
                "inverse_symbol_ops": 0.25,
                "direct_likelihood_ops": 1.0,
                "trellis_state_updates": 1.0,
                "trellis_traceback_ops": 0.5,
                "fallback_count": 0.0,
                "representation_atoms": 0.05,
            },
            "pessimistic": {
                "atoms_processed": 2.0,
                "generator_pushes": 1.0,
                "generator_pops": 2.0,
                "fiber_queries": 4.0,
                "fiber_solver_bitops": 0.1,
                "fiber_entries": 4.0,
                "score_updates": 2.0,
                "heap_pushes": 1.0,
                "heap_pops": 1.0,
                "membership_queries": 2.0,
                "inverse_symbol_ops": 0.5,
                "direct_likelihood_ops": 1.0,
                "trellis_state_updates": 2.0,
                "trellis_traceback_ops": 1.0,
                "fallback_count": 0.0,
                "representation_atoms": 0.2,
            },
        }
        if model not in weights:
            raise ValueError(f"Unknown cost model: {model}")
        d = asdict(self)
        return float(sum(d[k] * w for k, w in weights[model].items()))

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        for model in ("optimistic", "balanced", "pessimistic"):
            d[f"scalar_{model}"] = self.scalar(model)
        return d


@dataclass
class DecodeResult:
    decision: int
    ml_tie_set: tuple[int, ...]
    certified: bool
    exact: bool
    fallback_used: bool
    atoms_processed: int
    residual_mass: float
    work: WorkVector
    scores: np.ndarray | None = None
    notes: dict[str, Any] = field(default_factory=dict)

    def to_dict(self, include_scores: bool = False) -> dict[str, Any]:
        out: dict[str, Any] = {
            "decision": int(self.decision),
            "ml_tie_set": [int(v) for v in self.ml_tie_set],
            "certified": bool(self.certified),
            "exact": bool(self.exact),
            "fallback_used": bool(self.fallback_used),
            "atoms_processed": int(self.atoms_processed),
            "residual_mass": float(self.residual_mass),
            "work": self.work.to_dict(),
            "notes": self.notes,
        }
        if include_scores and self.scores is not None:
            out["scores"] = [float(v) for v in self.scores]
        return out
