import numpy as np

from ziv_gate.code_geometry import LinearCode, ambient_metric_arrays


def test_linear_code_closure():
    rng = np.random.default_rng(3)
    code = LinearCode.random_systematic(10, 6, rng)
    words = set(int(v) for v in code.codewords)
    assert len(words) == 64
    assert all((a ^ b) in words for a in words for b in words)


def test_ambient_metric_arrays():
    arrays = ambient_metric_arrays(8, include_ctw=True, ctw_depth=2)
    assert set(arrays) == {"LZ78_FIXED_BLOCK", "CTW_D2"}
    for metric in arrays.values():
        assert len(metric["raw"]) == 256
        assert metric["rank"].min() >= 0
        assert metric["rank"].max() <= 8 + 1e-9
