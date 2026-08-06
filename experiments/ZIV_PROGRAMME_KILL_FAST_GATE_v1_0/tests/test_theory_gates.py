from ziv_gate.individual_sequence import (
    deterministic_tie_counterexample,
    random_coding_formula_audit,
    regret_cancellation_counterexample,
)


def test_deterministic_tie_counterexample():
    witness = deterministic_tie_counterexample(8)
    assert witness["structural_failure"]
    assert witness["gap_per_symbol"] == 1.0


def test_random_coding_formula_audit():
    audit = random_coding_formula_audit()
    assert audit["cases_checked"] > 100
    assert audit["maximum_absolute_gap_proposal_power_vs_exact_without_replacement"] > 0


def test_regret_counterexample():
    witness = regret_cancellation_counterexample()
    assert witness["signed_total_regret_bits"] == 0
    assert witness["structural_failure"]
