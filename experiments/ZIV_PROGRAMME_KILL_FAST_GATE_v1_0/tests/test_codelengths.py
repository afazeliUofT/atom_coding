import itertools
import math

from ziv_gate.codelengths import (
    ctw_log2_probability,
    kt_log2_probability,
    lz78_codelength_bits,
    lz78_decode,
    lz78_records,
)


def test_lz_roundtrip_and_injectivity():
    seen = set()
    for n in range(9):
        for bits in itertools.product((0, 1), repeat=n):
            records = lz78_records(bits)
            assert lz78_decode(records, n) == bits
            key = (n, records)
            assert key not in seen
            seen.add(key)
            assert lz78_codelength_bits(bits) >= 2


def test_ctw_normalization():
    for depth in range(4):
        for n in range(1, 7):
            total = sum(2.0 ** ctw_log2_probability(bits, depth=depth) for bits in itertools.product((0, 1), repeat=n))
            assert math.isclose(total, 1.0, abs_tol=2e-11)


def test_kt_normalization():
    for order in range(3):
        for n in range(1, 7):
            total = sum(2.0 ** kt_log2_probability(bits, order=order) for bits in itertools.product((0, 1), repeat=n))
            assert math.isclose(total, 1.0, abs_tol=2e-11)
