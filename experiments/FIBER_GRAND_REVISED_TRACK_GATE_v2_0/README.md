# FIBER-GRAND Revised Track Gate v2.0

This is the decisive strong-baseline and theorem gate authorized after the first FIBER-GRAND kill-fast run.

## What it tests

1. Exact micro-world correctness and zero false certificates.
2. Codebook-free primary operation using syndrome or VT checksum membership.
3. Membership-only aggregate inverse search versus two exact code-specific baselines:
   - syndrome-trellis aggregate A*;
   - code-aware aggregate prefix A*.
4. Specialized VT one-deletion decoding.
5. Two-deletion transfer.
6. The deterministic fixed-edit shell-certificate theorem and its finite-length convergence toward the BSC `h2(p)` exponent.

## Authoritative classifications

- `CONTINUE_REVISED_FIBER_TRACK_TO_FINAL_GATE`
- `NARROW_TO_FIXED_EDIT_FOUNDATIONS_STOP_FIELD_DEFINING`
- `STOP_FIBER_FIELD_DEFINING_PROGRAMME`

A CONTINUE result authorizes only an independent novelty/proof audit, additional specialized synchronization-system baselines, and one calibrated post-front-end impairment. It does not establish field-defining status.

## Local execution

From the installed package root:

```bash
python -m fiber_v2.run_gate --profile standard
```

The external launcher creates the isolated virtual environment, runs tests, executes the standard profile, commits only this package, and pushes the active branch.

## Main outputs

```text
results/GATE_VERDICT.json
results/GATE_REPORT.md
results/02_primary_summary.csv
results/05_shell_theorem_gate.json
results/RESULTS_SHA256.txt
```

Exactly one command file is generated: `CONTINUE_COMMAND.txt`, `NARROW_COMMAND.txt`, or `STOP_COMMAND.txt`.
