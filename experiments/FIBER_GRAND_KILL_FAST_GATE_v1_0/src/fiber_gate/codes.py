from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np

from .utils import binary_tuple, int_array, tuple_to_int


def gf2_rank(matrix: np.ndarray) -> int:
    a = np.asarray(matrix, dtype=np.uint8).copy() % 2
    rows, cols = a.shape
    rank = 0
    for col in range(cols):
        pivot = next((row for row in range(rank, rows) if a[row, col]), None)
        if pivot is None:
            continue
        if pivot != rank:
            a[[rank, pivot]] = a[[pivot, rank]]
        for row in range(rows):
            if row != rank and a[row, col]:
                a[row] ^= a[rank]
        rank += 1
        if rank == rows:
            break
    return rank


def gf2_nullspace(matrix: np.ndarray) -> np.ndarray:
    a = np.asarray(matrix, dtype=np.uint8).copy() % 2
    rows, cols = a.shape
    pivot_cols: list[int] = []
    pivot_row = 0
    for col in range(cols):
        pivot = next((row for row in range(pivot_row, rows) if a[row, col]), None)
        if pivot is None:
            continue
        if pivot != pivot_row:
            a[[pivot_row, pivot]] = a[[pivot, pivot_row]]
        for row in range(rows):
            if row != pivot_row and a[row, col]:
                a[row] ^= a[pivot_row]
        pivot_cols.append(col)
        pivot_row += 1
        if pivot_row == rows:
            break
    pivot_set = set(pivot_cols)
    free_cols = [col for col in range(cols) if col not in pivot_set]
    basis = np.zeros((len(free_cols), cols), dtype=np.uint8)
    for basis_row, free in enumerate(free_cols):
        basis[basis_row, free] = 1
        for row, pivot_col in enumerate(pivot_cols):
            if a[row, free]:
                basis[basis_row, pivot_col] = 1
    return basis


def _systematic_generator_from_full_rank(generator: np.ndarray) -> np.ndarray:
    """Return a column-permuted systematic generator and the applied permutation.

    The gate keeps codewords in the resulting coordinate order, so no inverse
    permutation is needed in later experiments.
    """
    g = np.asarray(generator, dtype=np.uint8).copy() % 2
    k, n = g.shape
    if gf2_rank(g) != k:
        raise ValueError("Generator is not full row rank")
    pivot_row = 0
    column_order = list(range(n))
    for target_col in range(k):
        pivot = None
        pivot_col = None
        for col in range(target_col, n):
            candidate = next((row for row in range(pivot_row, k) if g[row, col]), None)
            if candidate is not None:
                pivot = candidate
                pivot_col = col
                break
        if pivot is None or pivot_col is None:
            raise AssertionError("Could not systematicize generator")
        if pivot_col != target_col:
            g[:, [target_col, pivot_col]] = g[:, [pivot_col, target_col]]
            column_order[target_col], column_order[pivot_col] = (
                column_order[pivot_col],
                column_order[target_col],
            )
        if pivot != pivot_row:
            g[[pivot_row, pivot]] = g[[pivot, pivot_row]]
        for row in range(k):
            if row != pivot_row and g[row, target_col]:
                g[row] ^= g[pivot_row]
        pivot_row += 1
    if not np.array_equal(g[:, :k], np.eye(k, dtype=np.uint8)):
        raise AssertionError("Systematicization failed")
    return g


@dataclass
class BinaryCode:
    name: str
    n: int
    codewords_int: np.ndarray
    codewords_array: np.ndarray
    family: str
    k_nominal: int | None = None

    def __post_init__(self) -> None:
        self.codewords_int = np.asarray(self.codewords_int, dtype=np.uint64)
        self.codewords_array = np.asarray(self.codewords_array, dtype=np.uint8)
        if self.codewords_array.shape != (len(self.codewords_int), self.n):
            raise ValueError("Codeword array has inconsistent dimensions")
        if len(np.unique(self.codewords_int)) != len(self.codewords_int):
            raise ValueError("Codewords must be distinct")
        self.membership: dict[int, int] = {
            int(word): int(index) for index, word in enumerate(self.codewords_int)
        }

    @property
    def size(self) -> int:
        return int(len(self.codewords_int))

    @property
    def rate(self) -> float:
        return math.log2(self.size) / self.n

    def message_index(self, word: int) -> int | None:
        return self.membership.get(int(word))

    def is_codeword(self, word: int) -> bool:
        return int(word) in self.membership

    def prefix_feasible(self, prefix: int, length: int) -> tuple[bool, int]:
        """L0 code: no generic prefix pruning; all prefixes remain possible."""
        return True, 0

    def metadata(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "family": self.family,
            "n": self.n,
            "size": self.size,
            "rate": self.rate,
            "k_nominal": self.k_nominal,
        }


class SystematicLinearCode(BinaryCode):
    def __init__(self, name: str, generator: np.ndarray, family: str = "linear") -> None:
        g = np.asarray(generator, dtype=np.uint8) % 2
        k, n = g.shape
        if not np.array_equal(g[:, :k], np.eye(k, dtype=np.uint8)):
            raise ValueError("SystematicLinearCode requires G=[I|P]")
        self.generator = g
        self.k = int(k)
        messages = ((np.arange(1 << k, dtype=np.uint64)[:, None] >> np.arange(k, dtype=np.uint64)) & 1).astype(np.uint8)
        words = (messages @ g) % 2
        powers = (np.uint64(1) << np.arange(n, dtype=np.uint64))[None, :]
        ints = np.sum(words.astype(np.uint64) * powers, axis=1, dtype=np.uint64)
        super().__init__(name, n, ints, words, family, k_nominal=k)

    @classmethod
    def random_systematic(
        cls,
        n: int,
        k: int,
        rng: np.random.Generator,
        name: str | None = None,
    ) -> "SystematicLinearCode":
        if not (1 <= k < n):
            raise ValueError("Require 1 <= k < n")
        parity = rng.integers(0, 2, size=(k, n - k), dtype=np.uint8)
        generator = np.concatenate([np.eye(k, dtype=np.uint8), parity], axis=1)
        return cls(name or f"RLC_n{n}_k{k}", generator, family="random_linear")

    def encode_message(self, message: int) -> int:
        return int(self.codewords_int[int(message)])

    def prefix_feasible(self, prefix: int, length: int) -> tuple[bool, int]:
        if length <= 0:
            return True, 0
        if length <= self.k:
            return True, 0
        message = int(prefix) & ((1 << self.k) - 1)
        expected = self.encode_message(message)
        mask = (1 << length) - 1
        compared = length - self.k
        return (expected & mask) == (int(prefix) & mask), max(1, compared)


def crc_systematic_code(n: int, k: int, polynomial: int, name: str | None = None) -> SystematicLinearCode:
    """Create a little-endian systematic CRC-like binary linear code.

    The polynomial includes its leading x^r coefficient. Message bits occupy
    positions 0..k-1 and parity bits positions k..n-1. The exact polynomial is
    frozen in the experiment metadata; the gate uses the code as an unmodified
    cyclic/CRC family rather than claiming standards compatibility.
    """
    r = n - k
    if polynomial.bit_length() != r + 1 or not (polynomial & 1):
        raise ValueError("Polynomial must have degree r and nonzero constant term")

    def remainder(message: int) -> int:
        value = int(message) << r
        divisor = int(polynomial)
        for shift in range(k - 1, -1, -1):
            bit_position = shift + r
            if (value >> bit_position) & 1:
                value ^= divisor << shift
        return value & ((1 << r) - 1)

    basis_words: list[int] = []
    for bit in range(k):
        message = 1 << bit
        parity = remainder(message)
        word = message | (parity << k)
        basis_words.append(word)
    generator = int_array(basis_words, n)
    if not np.array_equal(generator[:, :k], np.eye(k, dtype=np.uint8)):
        # The little-endian polynomial convention may not yield I|P directly;
        # systematicize while retaining a deterministic coordinate order.
        generator = _systematic_generator_from_full_rank(generator)
    return SystematicLinearCode(
        name or f"CRC_LIKE_n{n}_k{k}_poly{polynomial:x}",
        generator,
        family="crc_like",
    )


def hamming_15_11_code() -> SystematicLinearCode:
    h = np.zeros((4, 15), dtype=np.uint8)
    for col in range(15):
        value = col + 1
        for row in range(4):
            h[row, col] = (value >> row) & 1
    generator_basis = gf2_nullspace(h)
    generator = _systematic_generator_from_full_rank(generator_basis)
    return SystematicLinearCode("HAMMING_15_11", generator, family="hamming")


class VTCode(BinaryCode):
    def __init__(self, n: int, syndrome: int = 0) -> None:
        modulus = n + 1
        words: list[int] = []
        for value in range(1 << n):
            checksum = sum((index + 1) * ((value >> index) & 1) for index in range(n))
            if checksum % modulus == syndrome % modulus:
                words.append(value)
        array = int_array(words, n)
        super().__init__(
            f"VT_n{n}_a{syndrome % modulus}",
            n,
            np.asarray(words, dtype=np.uint64),
            array,
            family="varshamov_tenengolts",
            k_nominal=None,
        )
        self.syndrome = int(syndrome % modulus)

    def is_codeword(self, word: int) -> bool:
        checksum = sum((index + 1) * ((int(word) >> index) & 1) for index in range(self.n))
        return checksum % (self.n + 1) == self.syndrome

    def message_index(self, word: int) -> int | None:
        # Retain the dictionary for exact-baseline indexing while membership is
        # also independently checkable by the VT checksum.
        return self.membership.get(int(word))
