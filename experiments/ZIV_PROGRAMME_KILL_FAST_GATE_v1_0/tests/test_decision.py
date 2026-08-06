from ziv_gate.decision import PIVOT, STOP, final_decision


def base_inputs():
    exact = {"pass": True}
    individual = {
        "original_deterministic_conjecture": {"pass": False},
        "session_regret_transfer": {"pass_as_stated": False},
    }
    masking = {"combined": {"LZ78_FIXED_BLOCK": {"survives_falsification_only": True}}}
    stationary = {"pass": False}
    adaptation = {"pass": False}
    geometry = {"pass": False}
    traces = {"pass": False}
    return exact, individual, masking, stationary, adaptation, geometry, traces


def test_stop_when_flagship_and_geometry_fail():
    verdict = final_decision(*base_inputs())
    assert verdict["classification"] == STOP


def test_geometry_pivot_requires_both_signals():
    args = list(base_inputs())
    args[5] = {"pass": True}
    verdict = final_decision(*args)
    assert verdict["classification"] == PIVOT
    assert verdict["stage"] == "DESCRIPTION_LENGTH_GEOMETRY_ONLY"


def test_adaptive_pivot_is_narrow():
    args = list(base_inputs())
    args[3] = {"pass": True}
    args[4] = {"pass": True}
    verdict = final_decision(*args)
    assert verdict["classification"] == PIVOT
    assert verdict["stage"] == "ADAPTIVE_UNIVERSAL_GRAND_SYSTEMS_ONLY"
