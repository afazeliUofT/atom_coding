from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from typing import Iterator

import numpy as np


@dataclass
class ProductAtomItem:
    rank_state: tuple[int, ...]
    atom_indices: tuple[int, ...]
    probability: float
    log_probability: float


class ProductAtomEnumerator:
    """Best-first exact enumeration of i.i.d. product atoms by nonincreasing mass."""

    def __init__(self, probabilities: np.ndarray, blocklength: int) -> None:
        probs = np.asarray(probabilities, dtype=float)
        if probs.ndim != 1 or probs.size == 0:
            raise ValueError("probabilities must be a nonempty vector")
        if np.any(probs <= 0.0):
            raise ValueError("Product enumerator requires strictly positive atom probabilities")
        if not np.isclose(probs.sum(), 1.0, atol=1e-10):
            raise ValueError("Atom probabilities must sum to one")
        self.order = tuple(sorted(range(len(probs)), key=lambda i: (-float(probs[i]), i)))
        self.sorted_probs = np.array([probs[i] for i in self.order], dtype=float)
        self.costs = -np.log(self.sorted_probs / self.sorted_probs[0])
        self.base_log_probability = blocklength * math.log(float(self.sorted_probs[0]))
        self.blocklength = int(blocklength)
        self.support_size = len(probs)
        self.heap: list[tuple[float, tuple[int, ...]]] = []
        self.seen: set[tuple[int, ...]] = set()
        initial = (0,) * self.blocklength
        heapq.heappush(self.heap, (0.0, initial))
        self.seen.add(initial)
        self.pushes = 1
        self.pops = 0
        self.last_probability = float("inf")

    def __iter__(self) -> Iterator[ProductAtomItem]:
        return self

    def __next__(self) -> ProductAtomItem:
        if not self.heap:
            raise StopIteration
        cost, state = heapq.heappop(self.heap)
        self.pops += 1
        log_probability = self.base_log_probability - cost
        probability = math.exp(log_probability)
        if probability > self.last_probability * (1.0 + 1e-12):
            raise AssertionError("Product atom enumerator violated probability order")
        self.last_probability = min(self.last_probability, probability)

        for coordinate in range(self.blocklength):
            current = state[coordinate]
            if current + 1 >= self.support_size:
                continue
            neighbor = list(state)
            neighbor[coordinate] += 1
            neighbor_tuple = tuple(neighbor)
            if neighbor_tuple in self.seen:
                continue
            next_cost = cost + float(self.costs[current + 1] - self.costs[current])
            heapq.heappush(self.heap, (next_cost, neighbor_tuple))
            self.seen.add(neighbor_tuple)
            self.pushes += 1

        atom_indices = tuple(self.order[rank] for rank in state)
        return ProductAtomItem(
            rank_state=state,
            atom_indices=atom_indices,
            probability=probability,
            log_probability=log_probability,
        )
