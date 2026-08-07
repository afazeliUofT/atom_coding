# External Review Handoff - FIBER-GRAND Pre-Manuscript Closure v5

## Scope

This package does **not** ask whether generic path summation, A*, candidate-likelihood ordering, or conditional guesswork are new. Those claims are withdrawn. It asks whether the narrow fixed-edit theorem/decoder package is mathematically correct and whether the compiled performance evidence is strong enough to support a flagship IEEE Transactions on Information Theory manuscript.

## Required classification after a positive v5 run

Return exactly one of:

- `FLAGSHIP_TIT_THEORY_AND_ALGORITHM_CANDIDATE`
- `STRONG_BUT_NOT_FLAGSHIP_TIT_PAPER`
- `BLOCKING_MATHEMATICAL_DEFECT`
- `BLOCKING_NOVELTY_DEFECT`

## Review order

1. Read `FIBER_GRAND_Pre_Manuscript_Baseline_and_Tail_Closure.pdf`.
2. Verify the strict unseen-candidate certificate and complete-tie claim.
3. Verify the candidate-volume bounds and exact ambiguity degree.
4. Verify the arbitrary-code L0 lower bound, query sandwich, and known-cardinality refinement.
5. Verify the interior-shell exponent statement and its XP-not-FPT boundary.
6. Verify that moment equality is claimed only for actual-history revelation and only an upper bound for certified FIBER.
7. Inspect the C++20 FIBER, prefix A*, and syndrome-trellis implementations.
8. Confirm that the trellis is enabled at every required feasible point and all preprocessing is charged.
9. Review `03_error_weight_summary.csv` to ensure the signal survives positive substitution weight.
10. Audit the novelty matrix against primary sources and current patents/preprints.

## Hard-stop defects

- an unseen equal-likelihood codeword can survive the strict certificate;
- the known-cardinality oracle refinement is incorrect;
- the shell offsets in the L0 query sandwich are wrong;
- the candidate-volume bounds fail for some observation;
- a required feasible syndrome-trellis point is omitted or materially disadvantaged;
- preprocessing, failed searches, or memory allocation are not charged consistently;
- the positive-error-conditioned signal fails but the manuscript presents only unconditional timing;
- the narrow integrated theorem-and-decoder result is already present in a primary source.

## Main evidence

- `results/GATE_VERDICT.json`
- `results/01_candidate_theory_gate.json`
- `results/02_moment_tail_gate.json`
- `results/03_compiled_gate.json`
- `results/03_compiled_summary.csv`
- `results/03_error_weight_summary.csv`
- `results/03_compiled_trials.csv.gz`
- `results/05_vt_boundary_gate.json`
- `cpp/flagship_benchmark.cpp`
