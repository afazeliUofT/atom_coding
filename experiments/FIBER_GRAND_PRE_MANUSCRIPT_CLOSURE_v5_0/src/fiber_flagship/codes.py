from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from .utils import int_array


def gf2_rank(matrix: np.ndarray) -> int:
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


def gf2_nullspace(matrix: np.ndarray) -> np.ndarray:
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
    free_cols = [c for c in range(cols) if c not in set(pivot_cols)]
    basis = np.zeros((len(free_cols), cols), dtype=np.uint8)
    for br, free in enumerate(free_cols):
        basis[br, free] = 1
        for row, pivot_col in enumerate(pivot_cols):
            if a[row, free]:
                basis[br, pivot_col] = 1
    return basis


def systematicize(generator: np.ndarray) -> np.ndarray:
    g = np.asarray(generator, dtype=np.uint8).copy() % 2
    k, n = g.shape
    if gf2_rank(g) != k:
        raise ValueError("generator is not full rank")
    pivot_row = 0
    for target_col in range(k):
        pivot = None
        pivot_col = None
        for col in range(target_col, n):
            found = next((r for r in range(pivot_row, k) if g[r, col]), None)
            if found is not None:
                pivot, pivot_col = found, col
                break
        if pivot is None or pivot_col is None:
            raise AssertionError("systematicization failed")
        if pivot_col != target_col:
            g[:, [target_col, pivot_col]] = g[:, [pivot_col, target_col]]
        if pivot != pivot_row:
            g[[pivot_row, pivot]] = g[[pivot, pivot_row]]
        for r in range(k):
            if r != pivot_row and g[r, target_col]:
                g[r] ^= g[pivot_row]
        pivot_row += 1
    if not np.array_equal(g[:, :k], np.eye(k, dtype=np.uint8)):
        raise AssertionError("systematicization failed")
    return g


class CodeOracle(Protocol):
    name: str
    family: str
    n: int
    rate: float

    def is_codeword(self, word: int) -> tuple[bool, int]: ...
    def sample_codeword(self, rng: np.random.Generator) -> int: ...
    def metadata(self) -> dict[str, Any]: ...


@dataclass
class LinearOracle:
    name: str
    family: str
    generator: np.ndarray
    polynomial_name: str | None = None
    polynomial: int | None = None

    def __post_init__(self) -> None:
        g = np.asarray(self.generator, dtype=np.uint8) % 2
        if g.ndim != 2:
            raise ValueError("generator must be a matrix")
        self.generator = g
        self.k, self.n = g.shape
        if not np.array_equal(g[:, : self.k], np.eye(self.k, dtype=np.uint8)):
            raise ValueError("LinearOracle requires G=[I|P]")
        p = g[:, self.k :]
        self.parity_check = np.concatenate([p.T, np.eye(self.n - self.k, dtype=np.uint8)], axis=1)
        if np.any((g @ self.parity_check.T) % 2):
            raise AssertionError("G H^T != 0")
        self.row_masks = [sum(int(g[j, i]) << i for i in range(self.n)) for j in range(self.k)]
        self.syndrome_columns = [
            sum(int(self.parity_check[row, col]) << row for row in range(self.n - self.k))
            for col in range(self.n)
        ]

    @property
    def rate(self) -> float:
        return self.k / self.n

    @property
    def redundancy(self) -> int:
        return self.n - self.k

    @property
    def size(self) -> int:
        return 1 << self.k

    def encode(self, message: int) -> int:
        word = 0
        for j, mask in enumerate(self.row_masks):
            if (int(message) >> j) & 1:
                word ^= mask
        return word

    def sample_codeword(self, rng: np.random.Generator) -> int:
        return self.encode(int(rng.integers(0, 1 << self.k)))

    def syndrome(self, word: int) -> tuple[int, int]:
        syndrome = 0
        bitops = 0
        value = int(word)
        while value:
            lsb = value & -value
            position = lsb.bit_length() - 1
            syndrome ^= self.syndrome_columns[position]
            value ^= lsb
            bitops += max(1, self.redundancy)
        return syndrome, bitops

    def is_codeword(self, word: int) -> tuple[bool, int]:
        syndrome, bitops = self.syndrome(word)
        return syndrome == 0, bitops

    def enumerate_codewords(self, maximum: int = 1 << 20) -> np.ndarray:
        if self.size > maximum:
            raise ValueError("codebook too large to enumerate")
        return np.asarray([self.encode(m) for m in range(self.size)], dtype=np.uint64)

    def metadata(self) -> dict[str, Any]:
        payload = {
            "name": self.name,
            "family": self.family,
            "n": self.n,
            "k": self.k,
            "rate": self.rate,
            "size": self.size,
            "membership": "syndrome_only_no_codebook_table",
            "parity_check": self.parity_check.tolist(),
        }
        if self.polynomial is not None:
            payload["polynomial_hex"] = hex(self.polynomial)
            payload["polynomial_name"] = self.polynomial_name
            payload["bit_order_warning"] = "The named polynomial is used as a reproducible degree-r generator; this experiment does not claim wire-format compatibility with a specific standard implementation."
        return payload

    def prefix_feasible_systematic(self, prefix: int, length: int) -> tuple[bool, int]:
        if length <= self.k:
            return True, 0
        message = int(prefix) & ((1 << self.k) - 1)
        expected = self.encode(message)
        mask = (1 << length) - 1
        return (expected & mask) == (int(prefix) & mask), max(1, (length - self.k) * self.redundancy)


@dataclass
class VTOracle:
    n: int
    syndrome_value: int = 0

    def __post_init__(self) -> None:
        self.syndrome_value %= self.n + 1
        self.name = f"VT_n{self.n}_a{self.syndrome_value}"
        self.family = "varshamov_tenengolts"

    @property
    def rate(self) -> float:
        return max(0.0, 1.0 - math.log2(self.n + 1) / self.n)

    def checksum(self, word: int) -> tuple[int, int]:
        total = 0
        for i in range(self.n):
            total += (i + 1) * ((int(word) >> i) & 1)
        return total % (self.n + 1), self.n

    def is_codeword(self, word: int) -> tuple[bool, int]:
        checksum, ops = self.checksum(word)
        return checksum == self.syndrome_value, ops

    def sample_codeword(self, rng: np.random.Generator) -> int:
        if self.n > 63:
            # Build a random prefix then search a short suffix; avoids uint64 RNG limits.
            for _ in range(100000):
                word = int.from_bytes(rng.bytes((self.n + 7) // 8), "little") & ((1 << self.n) - 1)
                valid, _ = self.is_codeword(word)
                if valid:
                    return word
        else:
            for _ in range(100000):
                word = int(rng.integers(0, 1 << self.n, dtype=np.uint64))
                valid, _ = self.is_codeword(word)
                if valid:
                    return word
        raise RuntimeError("failed to sample VT codeword")

    def metadata(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "family": self.family,
            "n": self.n,
            "rate_proxy": self.rate,
            "membership": "direct_VT_checksum_no_codebook_table",
        }


def random_linear(n: int, k: int, rng: np.random.Generator, label: str = "") -> LinearOracle:
    parity = rng.integers(0, 2, size=(k, n - k), dtype=np.uint8)
    g = np.concatenate([np.eye(k, dtype=np.uint8), parity], axis=1)
    return LinearOracle(f"RLC_n{n}_k{k}{label}", "random_linear", g)


CRC_POLYNOMIALS: dict[int, tuple[int, str]] = {
    3: (0xB, "CRC-3/GSM-like"),
    4: (0x13, "CRC-4/ITU-like"),
    5: (0x25, "CRC-5/USB-like"),
    6: (0x67, "CRC-6/CDMA2000-A-like"),
    7: (0x89, "CRC-7/MMC-like"),
    8: (0x107, "CRC-8/ATM-like"),
    10: (0x633, "CRC-10/ATM-like"),
    12: (0x180F, "CRC-12/UMTS-like"),
    16: (0x11021, "CRC-16/CCITT-like"),
}


def crc_code(n: int, k: int, polynomial: int | None = None, label: str = "") -> LinearOracle:
    r = n - k
    if polynomial is None:
        polynomial, poly_name = CRC_POLYNOMIALS.get(
            r, ((1 << r) | (1 << max(1, r // 2)) | 1, f"deterministic-degree-{r}")
        )
    else:
        poly_name = f"user-degree-{r}"
    if polynomial.bit_length() != r + 1 or not (polynomial & 1):
        raise ValueError("polynomial must have degree r and odd constant term")

    def remainder(message: int) -> int:
        value = int(message) << r
        for shift in range(k - 1, -1, -1):
            if (value >> (shift + r)) & 1:
                value ^= int(polynomial) << shift
        return value & ((1 << r) - 1)

    rows: list[int] = []
    for bit in range(k):
        message = 1 << bit
        rows.append(message | (remainder(message) << k))
    g = int_array(rows, n)
    if not np.array_equal(g[:, :k], np.eye(k, dtype=np.uint8)):
        g = systematicize(g)
    return LinearOracle(
        f"CRC_n{n}_k{k}_poly{polynomial:x}{label}",
        "named_crc_linear",
        g,
        polynomial_name=poly_name,
        polynomial=polynomial,
    )


def hamming_15_11() -> LinearOracle:
    h = np.zeros((4, 15), dtype=np.uint8)
    for col in range(15):
        value = col + 1
        for row in range(4):
            h[row, col] = (value >> row) & 1
    g = systematicize(gf2_nullspace(h))
    return LinearOracle("HAMMING_15_11", "hamming", g)


def make_linear(family: str, n: int, k: int, rng: np.random.Generator, label: str = "") -> LinearOracle:
    normalized = family.upper()
    if normalized == "RLC":
        return random_linear(n, k, rng, label)
    if normalized == "CRC":
        return crc_code(n, k, label=label)
    if normalized == "HAMMING":
        if (n, k) != (15, 11):
            raise ValueError("Hamming family supports only (15,11)")
        return hamming_15_11()
    raise ValueError(f"unknown linear family {family}")
