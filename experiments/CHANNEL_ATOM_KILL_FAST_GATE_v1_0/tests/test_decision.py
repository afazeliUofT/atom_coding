from __future__ import annotations

from atom_gate.decision import CONTINUE, PIVOT, STOP, early_decision, final_decision


def test_exactness_failure_is_immediate_stop() -> None:
    verdict = early_decision(exactness={"pass": False})
    assert verdict is not None
    assert verdict["classification"] == STOP
    assert verdict["stage"] == "H1_EXACTNESS"


def test_absent_atlas_mechanism_is_stop() -> None:
    verdict = early_decision(
        exactness={"pass": True},
        atlas={
            "exact_all": True,
            "nonadditive_variation_witnesses": 0,
            "nonadditive_natural_improvement_witnesses": 0,
        },
    )
    assert verdict is not None
    assert verdict["classification"] == STOP


def test_two_joint_nonlatin_families_authorize_only_provisional_continue() -> None:
    exactness = {"pass": True}
    atlas = {
        "exact_all": True,
        "nonadditive_variation_witnesses": 4,
        "nonadditive_natural_improvement_witnesses": 2,
    }
    reversible = {
        "pass": True,
        "pass_flags": {"NONCYCLIC_REVERSIBLE_ACTION_q5": True},
    }
    nonlatin = {"channel_pass": {"A": True, "B": True}, "passing_channels": 2}
    rate = {"channel_pass": {"A": True, "B": True}, "passing_channels": 2}
    verdict = final_decision(exactness, atlas, reversible, nonlatin, rate)
    assert verdict["classification"] == CONTINUE
    assert "novelty" in verdict["warning"].lower()


def test_reversible_only_is_pivot_not_continue() -> None:
    exactness = {"pass": True}
    atlas = {
        "exact_all": True,
        "nonadditive_variation_witnesses": 4,
        "nonadditive_natural_improvement_witnesses": 2,
    }
    reversible = {
        "pass": True,
        "pass_flags": {"NONCYCLIC_REVERSIBLE_ACTION_q5": True},
    }
    nonlatin = {"channel_pass": {"A": False}, "passing_channels": 0}
    rate = {"channel_pass": {"A": False}, "passing_channels": 0}
    verdict = final_decision(exactness, atlas, reversible, nonlatin, rate)
    assert verdict["classification"] == PIVOT
