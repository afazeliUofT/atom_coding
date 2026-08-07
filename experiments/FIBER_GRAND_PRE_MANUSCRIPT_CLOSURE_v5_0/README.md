# FIBER-GRAND Pre-Manuscript Closure Gate v5.0

This package is a **mandatory closure gate** between commit `2f67a32` and a full IEEE Transactions on Information Theory manuscript.

The prior v4 run passed its frozen decision contract, reproduced exactness, established a narrow oracle-relative theorem, and showed a large same-toolchain compiled advantage. Independent review nevertheless found two paper-readiness weaknesses:

1. the syndrome-trellis exact baseline was disabled at several larger blocklength/rate points where its charged dynamic-programming cost was feasible; and
2. the largest-block `p95`/`p99` summaries were based on only 12 trials, while the low substitution probabilities made zero-substitution blocks the majority in several configurations.

The v5 gate does not alter the channel, decoder, theorem, or claim boundary. It asks whether the performance claim survives a fairer exact reference and statistically supported positive-error conditioning.

## Frozen narrow claim

For a fixed number of uniformly located deletions followed by BSC substitutions on the surviving symbols, membership-first aggregate inverse search returns the complete ML tie set using a strict unseen-candidate certificate. In an arbitrary-code L0 membership-oracle model, its distinct-candidate queries are polynomially competitive with the optimal exact L0 query strategy for interior mismatch shells. Fixed-edit history revelation has Renyi moment and large-deviation laws, while certified FIBER inherits the corresponding upper bounds.

## New closure requirements

The standard profile:

- preserves all exactness and theorem audits from v4;
- enables the syndrome-trellis baseline at all required feasible points:
  - `n=48, R=0.75`, estimated `18,874,368` charged DP updates per block;
  - `n=48, R=0.875`, estimated `294,912` updates;
  - `n=64, R=0.875`, estimated `2,097,152` updates;
- keeps `n=64, R=0.75` outside the mandatory trellis set because the estimate is `536,870,912` updates per block;
- uses 500+ largest-block trials per key configuration;
- performs three timing repeats after per-configuration warmup;
- randomizes decoder execution order;
- reports `E=0`, `E=1`, `E=2`, and `E>=3` strata separately;
- requires positive-substitution evidence rather than accepting a signal created only by `E=0` blocks;
- uses supported `p99` summaries and confidence intervals;
- retains the true O(n) VT decoder as an honest specialization boundary.

## Possible scientific classifications

- `AUTHORIZE_FULL_TIT_MANUSCRIPT_AND_EXTERNAL_REVIEW`
- `NARROW_TO_FIXED_EDIT_TIT_THEORY_ALGORITHM_PAPER`
- `STOP_FLAGSHIP_PERFORMANCE_CLAIM`

A positive result authorizes manuscript construction and independent review. It does not establish field-defining impact, hardware superiority, FPT dependence on edit count, or real-system relevance.

## Run inside the installed package

```bash
python RUN_GATE.py --profile standard
```

The user-facing wrapper performs hash verification, safe extraction, isolated-environment setup, tests, C++ compilation, scientific execution, Git commit, and Git push.

## Principal outputs

- `results/GATE_VERDICT.json`
- `results/GATE_REPORT.md`
- `results/ANALYSIS_HANDOFF.md`
- `results/01_candidate_theory_gate.json`
- `results/03_compiled_gate.json`
- `results/03_error_weight_summary.csv`
- `results/RESULTS_SHA256.txt`
- `theory/FIBER_GRAND_Pre_Manuscript_Baseline_and_Tail_Closure.pdf`
