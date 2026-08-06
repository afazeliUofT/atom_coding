# Ziv Programme Kill-Fast Gate v1.0

This package tests whether the proposal **Universal Guessing Decoders for Machine-Generated Interference: An Individual-Sequence Theory of Decoding and the Description-Length Geometry of Codes** warrants long-term investment as a field-defining coding programme.

It is intentionally adversarial. It freezes the proposal's load-bearing claims, searches for exact counterexamples before running performance experiments, charges absolute query complexity, compares structured errors against matched controls, and emits exactly one investment command.

## Authoritative reference result

The release-standard validation run returns:

```text
STOP_FIELD_DEFINING_ZIV_PROGRAMME
STAGE=FLAGSHIP_AND_GEOMETRY_COLLAPSE
```

The WSL run independently reproduces that classification in the user's environment. More computation is not authorized to rescue a failed claim. The `deep` profile is blocked by default.

## Collapse propositions

- **Z0 - implementation and normalization:** all codelength, probability, type-count, ensemble-law, and code computations must pass exact audits.
- **Z1 - deterministic individual-sequence theorem:** the pointwise rank theorem must survive unrestricted ties, or be replaced by a genuinely new tie-invariant theorem.
- **Z2 - reliability transfer:** the claimed regret-to-error implication must survive exponentiation, saturation, and averaging.
- **Z3 - effective XOR geometry:** one frozen practical codelength must satisfy the required masking inequalities without linear defect.
- **Z4 - practical stationary ranking:** a universal ranker must remain close to the oracle, beat memoryless/fitted alternatives where structure exists, and remain inside absolute query budgets.
- **Z5 - nonstationary advantage:** online adaptation must beat a cheap fixed fitted model under abrupt drift.
- **Z6 - code-design signal:** description-length-selected codes must correct structured errors more often than matched-weight controls at useful finite blocklengths.
- **Z7 - systems premise:** real post-front-end traces from at least two domains must exhibit query-feasible Renyi-1/2 rates.
- **Z8 - novelty:** the surviving theorem must not already be contained in universal decoding, randomized individual-sequence guessing, porosity, or universal GRAND literature.

## Computational stages

1. `E0`: exactness and normalization audit.
2. `E1`: deterministic tie counterexample, exact random-code ensemble law, signed-regret counterexample, repaired level-set rank atlas.
3. `E2`: exact and heuristic XOR-masking counterexample search for a frozen LZ78 encoder and CTW depths 1/2.
4. `E3`: exact Markov-type level-set rank benchmark for oracle, KT universal, target-fitted, short-fit, long-fit, and memoryless models.
5. `E4`: online adaptation under abrupt switching.
6. `E5`: finite-block random-linear code geometry and structured-versus-matched-control correction tests.
7. `E6`: synthetic entropy/Renyi audit and optional real-trace ingestion.

## Profiles

- `smoke`: installation and implementation smoke test only.
- `standard`: authoritative preregistered investment gate.
- `deep`: blocked by default. It may not be run unless a new claim contract is frozen after a legitimate surviving standard result.

Run directly from the extracted package with:

```bash
python RUN_GATE.py --profile standard
```

The external launcher supplied with the release performs installation, testing, execution, hashing, Git staging, commit, and push.

## Real trace format

Optional real traces belong in:

```text
data/real_traces/
```

Accepted files are plain text or CSV-like files containing binary hard-decision impairment/error symbols. Non-binary tokens are rejected. A real-trace pass is not required to establish the current STOP because the core theorem and geometry gates already fail; it is required before any future systems-only claim.

## Decisive result files

```text
results/GATE_VERDICT.json
results/GATE_REPORT.md
results/ANALYSIS_HANDOFF.md
results/01_exactness_audit.json
results/02_individual_sequence_gate.json
results/03_masking_gate.json
results/04_stationary_rank_gate.json
results/05_nonstationary_gate.json
results/06_code_geometry_gate.json
results/07_trace_gate.json
results/RESULTS_SHA256.txt
results/STOP_COMMAND.txt
```

Detailed CSV/GZIP trial tables and figures are retained so all summaries can be independently recomputed.

## Interpretation discipline

- A finite counterexample refutes a universal theorem.
- Failure to find a counterexample does **not** prove a universal theorem.
- Relative closeness to an oracle does not establish practical feasibility when the absolute query count violates the frozen budget.
- LZ/CTW log loss estimates Shannon-type compressibility; it is not automatically a Renyi-1/2 estimator.
- An installation, authentication, or environment error must be reported as `EXECUTION_BLOCKED_NOT_A_SCIENTIFIC_VERDICT`, never as a scientific STOP.
