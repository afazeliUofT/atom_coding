import numpy as np

from fiber_final.codes import VTOracle, crc_code, random_linear


def test_random_linear_syndrome_closure():
    rng = np.random.default_rng(1)
    code = random_linear(12, 9, rng)
    for m in range(1 << code.k):
        assert code.is_codeword(code.encode(m))[0]


def test_crc_syndrome_closure():
    code = crc_code(12, 9)
    for m in range(1 << code.k):
        assert code.is_codeword(code.encode(m))[0]


def test_noncode_syndrome_detected():
    rng = np.random.default_rng(2)
    code = random_linear(12, 9, rng)
    word = code.encode(7)
    assert not code.is_codeword(word ^ (1 << 10))[0] or not code.is_codeword(word ^ (1 << 11))[0]


def test_vt_checksum():
    rng = np.random.default_rng(3)
    code = VTOracle(11, 0)
    for _ in range(10):
        word = code.sample_codeword(rng)
        assert code.is_codeword(word)[0]
