# FIBER-GRAND Kill-Fast Scientific Gate v1.0

This package tests whether the FIBER-GRAND proposal has an evidence-based route to a field-defining coding/decoding programme before a multi-year investment.

The package is deliberately stricter than a correctness demo. It separates:

1. exactness and false-certificate risk;
2. whether path multiplicity actually changes ML decisions;
3. one-deletion computational advantage;
4. search inflation and history duplication;
5. transfer to one insertion and two deletions;
6. transfer across unmodified code families;
7. the mandatory novelty and strong-baseline gates that remain outside the initial computation.

## Scientific boundary

The package treats the proposal's aggregate likelihood semantics as correct. It also records two necessary claim corrections:

- `fiber guesswork` under a uniform input prior is ordinary posterior conditional guesswork with side information;
- a residual/frontier stopping rule is a branch-and-bound certificate, not by itself a new information-theoretic theorem.

A positive standard run therefore returns a **revised pivot**, not an unconditional field-defining continuation.

## Profiles

- `smoke`: implementation check only; not an investment verdict.
- `standard`: authoritative early investment gate.
- `deep`: blocked. It requires a new frozen claim contract and must not be used to rescue a failed hypothesis.

## Direct package execution

From an installed package and activated environment:

```bash
python -m fiber_gate.run_gate --profile standard
```

or

```bash
python RUN_GATE.py --profile standard
```

## Stages

- **E0**: exact likelihood, certificate, and code-interface audit.
- **E1**: conditional-guesswork identity, random-code rank law, and multiplicity counterexample.
- **E2**: pathwise first-hit disagreement and search-inflation audit.
- **E3**: one deletion plus substitutions, random-linear and CRC-like codes.
- **E4**: one insertion plus substitutions.
- **E5**: two deletions plus substitutions.
- **E6**: random-linear, CRC-like, Hamming, and VT code transfer.

## Algorithms

- `HISTORY_L0`: membership-only, probability-ordered inverse histories; exact score only when a codeword is discovered.
- `PREFIX_L0`: code-independent aggregate prefix bound.
- `PREFIX_L2`: systematic-linear prefix feasibility; reported separately as code-assisted acceleration.
- `EXHAUSTIVE_CODEWORD_ML`: independent exact oracle and initial work baseline.

## Cost discipline

The raw work vector includes history components, distinct candidates, duplicate histories, membership queries, exact-score calls, symbol operations, candidate construction, heap operations, bound checks, prefix nodes, prefix-feasibility work, frontier sizes, memory, fallback, and wall-clock time.

The frozen `optimistic`, `balanced`, and `pessimistic` scalarizations are convenience summaries only. Raw per-trial tables are retained so reviewers can apply another cost model.

The operation-count speedups are comparisons with exhaustive codeword-wise likelihood evaluation. They are **not** claimed as equivalent wall-clock or hardware speedups. Strong priority-first, code-trellis, and specialized insertion/deletion baselines are mandatory in the next contract.

## Authoritative outputs

Read in this order:

1. `results/GATE_VERDICT.json`
2. `results/GATE_REPORT.md`
3. `results/01_exactness_audit.json`
4. `results/02_theory_boundary.json`
5. `results/03_boundary_audit.json`
6. `results/04_one_deletion_scaling.json`
7. `results/05_one_insertion_scaling.json`
8. `results/06_two_deletion_scaling.json`
9. `results/07_code_transfer.json`
10. `results/RESULTS_SHA256.txt`

Exactly one of `STOP_COMMAND.txt`, `PIVOT_COMMAND.txt`, or `CONTINUE_COMMAND.txt` is authoritative for investment discipline.

## Reference standard outcome

The release-validation standard run returns:

```text
PIVOT_TO_REVISED_FIBER_TRACK
```

This means:

```text
STOP_ORIGINAL_FIBER_GRAND_CLAIM_SET
```

The computational core is promising, but the original claim set must be revised. The only authorized next path is an independent novelty matrix, strong exact/specialized baselines, extraction of a new complexity theorem, and one calibrated use case.

## What the result does not establish

It does not establish:

- field-defining status;
- novelty of conditional guesswork;
- asymptotic work superiority;
- polynomial-time exact decoding for general weighted transducers;
- superiority over VT/GC+/marker/polar synchronization systems;
- a real telecommunications advantage.
