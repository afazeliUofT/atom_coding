# Analysis Handoff

Use this file to review the run without searching the repository.

## Read in this order

1. `GATE_VERDICT.json` — final machine classification and compact evidence.
2. `GATE_REPORT.md` — numerical interpretation and stop/pivot boundary.
3. `01_exactness_audit.json` — independent correctness checks.
4. `02_small_dmc_atlas_summary.json` and `02_small_dmc_channel_summary.csv` — H2.
5. `04_reversible_action_scaling.json` and `04_reversible_action_summary.csv` — positive controls.
6. `05_nonlatin_scaling.json` and `05_nonlatin_summary.csv` — decisive H3 tests.
7. `06_rate_survival.json`, `06_rate_survival_analytic.csv`, and `06_rate_survival_graph.csv` — H4.
8. `RESULTS_SHA256.txt` — integrity manifest.

Exactly one of `STOP_COMMAND.txt`, `PIVOT_COMMAND.txt`, or `CONTINUE_COMMAND.txt` is authoritative for investment discipline.

The detailed compressed trial tables are retained so that means, tails, completion fractions, and fitted slopes can be recomputed independently.
