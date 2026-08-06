from __future__ import annotations

from fiber_gate.decision import STOP, final_decision


def test_exactness_failure_forces_stop() -> None:
    verdict, _ = final_decision(
        {"pass": False},
        {"pass": True},
        {"results": {}, "exact_all": True},
        {"representations": {}, "exact_all": True},
        {"representations": {}, "exact_all": True},
        {"representations": {}, "exact_all": True},
        {"results": {}, "exact_all": True},
    )
    assert verdict["classification"] == STOP
