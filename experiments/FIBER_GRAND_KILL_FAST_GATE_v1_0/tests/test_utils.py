from __future__ import annotations

from fiber_gate.utils import binary_tuple, tuple_to_int


def test_binary_roundtrip() -> None:
    for n in range(1, 17):
        for value in range(min(1 << n, 100)):
            assert tuple_to_int(binary_tuple(value, n)) == value
