# Release Validation - Ziv Programme Kill-Fast Gate v1.0

## Scope

This validation checks package integrity, implementation tests, complete execution of the frozen `standard` profile, result reproducibility, and PDF usability. It does not convert finite tests into a universal impossibility theorem and does not substitute for independent primary-source novelty adjudication.

## Environment

The reference run used the container Python environment recorded in:

```text
validation/reference_standard_run/00_environment.json
```

The user's WSL launcher creates a fresh project-local virtual environment and records its own environment snapshot.

## Test result

```text
13 passed
```

See `TEST_REPORT.txt`.

## Standard scientific run

All stages E0-E6 completed with exit code zero. Runtime and peak resident memory are retained in `REFERENCE_EXECUTION_LOG.txt`.

Reference classification:

```text
STOP_FIELD_DEFINING_ZIV_PROGRAMME
STAGE=FLAGSHIP_AND_GEOMETRY_COLLAPSE
```

The decisive failures are scientific rather than operational:

- exact deterministic tie counterexample to the original pointwise theorem;
- exact counterexample to signed-regret cancellation as an average-error guarantee;
- explicit CTW XOR-masking violations;
- no useful frozen short-block description-length code-geometry signal;
- universal stationary ranking outside the proposal's absolute query budget;
- no adaptive advantage over the fixed fitted benchmark in the frozen switching test;
- real post-front-end trace premise unclosed.

The frozen LZ78 codelength survives finite masking falsification only. That absence of a witness is explicitly not treated as a theorem or a continuation signal.

## PDF validation

- 12 pages, US Letter.
- Searchable, not encrypted, not scanned.
- Fonts embedded.
- Outline present.
- Rendered successfully at 180 dpi.
- Representative pages 1, 6, and 12 visually inspected with no clipping, overlap, black boxes, or broken equations.

See `pdf_validation/`.

## Launcher rehearsal

A clean temporary-repository rehearsal verified safe installation, venv setup, all tests, the complete standard run, STOP command emission, and Git commit creation with launcher exit code zero. Only the remote push was intentionally skipped. See `LAUNCHER_REHEARSAL.txt`.

## Investment discipline

The `deep` profile is blocked by default. A failed standard gate must not be rescued by adding computation or changing thresholds after the fact.

An installation, authentication, or Git push failure in WSL is classified as:

```text
EXECUTION_BLOCKED_NOT_A_SCIENTIFIC_VERDICT
```

It cannot overwrite or fabricate a scientific conclusion.
