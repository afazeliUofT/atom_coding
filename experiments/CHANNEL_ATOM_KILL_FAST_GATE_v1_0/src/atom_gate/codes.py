from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable

import numpy as np



def int_to_digits(value: int, length: int, base: int) -> tuple[int, ...]:
    digits = [0] * length
    current = int(value)
    for i in range(length):
        digits[i] = current % base
        current //= base
    return tuple(digits)


def digits_to_int(digits: Iterable[int], base: int) -> int:
    value = 0
    multiplier = 1
    for digit in digits:
        value += int(digit) * multiplier
        multiplier *= base
    return value


def binary_int_to_array(value: int, length: int) -> np.ndarray:
    return np.fromiter(((value >> i) & 1 for i in range(length)), dtype=np.uint8, count=length)


@dataclass
class FiberSolution:
    message_indices: np.ndarray
    rank: int
    nullity: int
    consistent: bool
    estimated_bitops: int


class BinaryLinearCode:
    """Small exact binary linear code with a GF(2) coordinate-constraint fiber oracle."""

    def __init__(self, generator: np.ndarray, name: str = "binary_linear") -> None:
        g = np.asarray(generator, dtype=np.uint8) % 2
        if g.ndim != 2:
            raise ValueError("Generator must be a matrix")
        self.generator = g
        self.k, self.n = g.shape
        if self.k > self.n:
            raise ValueError("Code dimension cannot exceed blocklength")
        if self._gf2_rank(g) != self.k:
            raise ValueError("Generator matrix is not full row rank")
        self.name = name
        self.size = 1 << self.k
        self.row_masks = [sum(int(g[j, i]) << i for i in range(self.n)) for j in range(self.k)]
        self.column_masks = [sum(int(g[j, i]) << j for j in range(self.k)) for i in range(self.n)]
        self.codewords_int = np.zeros(self.size, dtype=np.uint64)
        for message in range(self.size):
            word = 0
            for j in range(self.k):
                if (message >> j) & 1:
                    word ^= self.row_masks[j]
            self.codewords_int[message] = word
        self.codewords_array = np.vstack(
            [binary_int_to_array(int(word), self.n) for word in self.codewords_int]
        )
        self.membership = {int(word): int(index) for index, word in enumerate(self.codewords_int)}
        self.parity_check = self._gf2_nullspace(g)
        if self.parity_check.shape != (self.n - self.k, self.n):
            raise AssertionError("Parity-check construction returned the wrong shape")
        if self.parity_check.size and np.any((g @ self.parity_check.T) % 2):
            raise AssertionError("Constructed parity-check matrix is not orthogonal to the generator")
        self.syndrome_column_masks = [
            sum(int(self.parity_check[row, col]) << row for row in range(self.n - self.k))
            for col in range(self.n)
        ]
        self._fiber_cache: dict[tuple[int, int], FiberSolution] = {}

    @staticmethod
    def _gf2_rank(matrix: np.ndarray) -> int:
        a = np.asarray(matrix, dtype=np.uint8).copy() % 2
        rows, cols = a.shape
        rank = 0
        for col in range(cols):
            pivot = next((r for r in range(rank, rows) if a[r, col]), None)
            if pivot is None:
                continue
            if pivot != rank:
                a[[rank, pivot]] = a[[pivot, rank]]
            for r in range(rows):
                if r != rank and a[r, col]:
                    a[r] ^= a[rank]
            rank += 1
            if rank == rows:
                break
        return rank

    @staticmethod
    def _gf2_nullspace(matrix: np.ndarray) -> np.ndarray:
        """Return a row-basis for {h : matrix @ h^T = 0} over GF(2)."""
        a = np.asarray(matrix, dtype=np.uint8).copy() % 2
        rows, cols = a.shape
        pivot_cols: list[int] = []
        pivot_row = 0
        for col in range(cols):
            pivot = next((r for r in range(pivot_row, rows) if a[r, col]), None)
            if pivot is None:
                continue
            if pivot != pivot_row:
                a[[pivot_row, pivot]] = a[[pivot, pivot_row]]
            for r in range(rows):
                if r != pivot_row and a[r, col]:
                    a[r] ^= a[pivot_row]
            pivot_cols.append(col)
            pivot_row += 1
            if pivot_row == rows:
                break
        free_cols = [col for col in range(cols) if col not in set(pivot_cols)]
        basis = np.zeros((len(free_cols), cols), dtype=np.uint8)
        for basis_row, free in enumerate(free_cols):
            basis[basis_row, free] = 1
            for row, pivot_col in enumerate(pivot_cols):
                if a[row, free]:
                    basis[basis_row, pivot_col] = 1
        return basis

    @classmethod
    def random_systematic(
        cls,
        n: int,
        k: int,
        rng: np.random.Generator,
        name: str | None = None,
    ) -> "BinaryLinearCode":
        if not (1 <= k <= n):
            raise ValueError("Require 1 <= k <= n")
        parity = rng.integers(0, 2, size=(k, n - k), dtype=np.uint8)
        generator = np.concatenate([np.eye(k, dtype=np.uint8), parity], axis=1)
        return cls(generator, name=name or f"RLC_n{n}_k{k}")

    def encode(self, message: int) -> int:
        return int(self.codewords_int[message])

    def message_index(self, word: int) -> int | None:
        return self.membership.get(int(word))

    def fiber(self, fixed_mask: int, fixed_value: int) -> FiberSolution:
        key = (int(fixed_mask), int(fixed_value & fixed_mask))
        cached = self._fiber_cache.get(key)
        if cached is not None:
            return cached

        equations: list[list[int]] = []
        for coordinate in range(self.n):
            if (fixed_mask >> coordinate) & 1:
                coeff = self.column_masks[coordinate]
                rhs = (fixed_value >> coordinate) & 1
                equations.append([coeff, rhs])

        rows = len(equations)
        pivot_cols: list[int] = []
        pivot_row = 0
        bitops = 0
        for col in range(self.k):
            pivot = next((r for r in range(pivot_row, rows) if (equations[r][0] >> col) & 1), None)
            if pivot is None:
                continue
            if pivot != pivot_row:
                equations[pivot_row], equations[pivot] = equations[pivot], equations[pivot_row]
            for r in range(rows):
                if r != pivot_row and ((equations[r][0] >> col) & 1):
                    equations[r][0] ^= equations[pivot_row][0]
                    equations[r][1] ^= equations[pivot_row][1]
                    bitops += self.k + 1
            pivot_cols.append(col)
            pivot_row += 1
            if pivot_row == rows:
                break

        for coeff, rhs in equations:
            if coeff == 0 and rhs:
                solution = FiberSolution(
                    message_indices=np.empty(0, dtype=np.int64),
                    rank=len(pivot_cols),
                    nullity=self.k - len(pivot_cols),
                    consistent=False,
                    estimated_bitops=bitops + rows * max(1, self.k),
                )
                self._fiber_cache[key] = solution
                return solution

        pivot_to_row = {col: r for r, col in enumerate(pivot_cols)}
        free_cols = [col for col in range(self.k) if col not in pivot_to_row]

        particular = 0
        for col, r in pivot_to_row.items():
            if equations[r][1] & 1:
                particular |= 1 << col

        basis: list[int] = []
        for free in free_cols:
            vector = 1 << free
            for pivot_col, r in pivot_to_row.items():
                if (equations[r][0] >> free) & 1:
                    vector |= 1 << pivot_col
            basis.append(vector)

        count = 1 << len(basis)
        indices = np.empty(count, dtype=np.int64)
        for combination in range(count):
            value = particular
            for j, vector in enumerate(basis):
                if (combination >> j) & 1:
                    value ^= vector
            indices[combination] = value

        solution = FiberSolution(
            message_indices=indices,
            rank=len(pivot_cols),
            nullity=len(free_cols),
            consistent=True,
            estimated_bitops=bitops + rows * max(1, self.k) + count * max(1, len(basis)),
        )
        self._fiber_cache[key] = solution
        return solution

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "n": self.n,
            "k": self.k,
            "rate": self.k / self.n,
            "generator": self.generator.tolist(),
            "parity_check": self.parity_check.tolist(),
        }


class QaryRandomCodebook:
    def __init__(self, q: int, n: int, words: Iterable[int], name: str = "qary_random") -> None:
        self.q = int(q)
        self.n = int(n)
        unique = sorted(set(int(word) for word in words))
        if not unique:
            raise ValueError("Codebook cannot be empty")
        ambient = self.q**self.n
        if unique[0] < 0 or unique[-1] >= ambient:
            raise ValueError("Codeword integer outside q-ary ambient space")
        self.words = np.asarray(unique, dtype=np.int64)
        self.size = len(unique)
        self.name = name
        self.membership = {int(word): i for i, word in enumerate(unique)}
        self.digits = np.vstack([np.array(int_to_digits(int(word), self.n, self.q), dtype=np.int16) for word in unique])

    @classmethod
    def random(
        cls,
        q: int,
        n: int,
        size: int,
        rng: np.random.Generator,
        name: str | None = None,
    ) -> "QaryRandomCodebook":
        ambient = q**n
        if not (1 <= size <= ambient):
            raise ValueError("Invalid code size")
        selected: set[int] = set()
        while len(selected) < size:
            batch = rng.integers(0, ambient, size=max(64, 2 * (size - len(selected))), dtype=np.int64)
            selected.update(int(v) for v in batch)
        words = list(selected)[:size]
        return cls(q, n, words, name or f"QARY_RANDOM_q{q}_n{n}_M{size}")

    def message_index(self, word: int) -> int | None:
        return self.membership.get(int(word))

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "q": self.q,
            "n": self.n,
            "size": self.size,
            "rate_bits_per_symbol": np.log2(self.size) / self.n,
            "normalized_rate": np.log(self.size) / (self.n * np.log(self.q)),
        }
