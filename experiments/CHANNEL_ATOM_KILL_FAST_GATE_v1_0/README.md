# Channel-Atom Coding Kill-Fast Gate v1.0

This package is the first scientific investment gate for the Channel-Atom Coding proposal. It is deliberately designed to falsify the broad programme before work begins on asymptotic coding theorems, code-family construction, finite-state channels, synchronization channels, hardware, or applications.

The package contains:

- `theory/Channel_Atom_Collapse_Propositions_and_Kill_Fast_Protocol.pdf`: the developed theoretical dossier;
- `theory/Channel_Atom_Collapse_Propositions_and_Kill_Fast_Protocol.tex`: its LaTeX source;
- `src/atom_gate/`: exact representation, decoding, linear-programming, graph, scaling, and verdict code;
- `config/profiles.json`: smoke, standard, and deep computational profiles;
- `config/decision_contract.json`: frozen numerical thresholds;
- `tests/`: independent implementation tests;
- `theory/MANUAL_NOVELTY_GATE.md`: the non-automatable H0 prior-art boundary;
- `validation/`: release-validation evidence;
- `results/`: the machine-generated evidence and final command after execution.

## The five load-bearing gates

1. **H0 - novelty boundary.** The operational optimization over exact latent deterministic representations for certified ML work and code-restricted fibers must not already be established. This is a manual primary-source audit.
2. **H1 - exactness.** Residual-certified atom decoding, product enumeration, decomposition LPs, GF(2) fibers, and exact references must agree with independent exact calculations.
3. **H2 - representation dependence.** Reduced exact representations of the same nonadditive channel must produce a real, fully charged work difference and improve over natural couplings.
4. **H3 - scaling.** At least two structurally distinct non-Latin channel families must beat the strongest implemented exact reference with favorable work slopes and no pessimistic-cost reversal.
5. **H4 - rate survival.** Fiber growth and likely-atom confusability must leave a nontrivial fraction of the relevant information rate.

Failure at H1, H2, or the positive-control stage causes an immediate machine STOP. Failure of broad H3/H4 can either stop the broad programme or authorize only a narrowly specified pivot. A positive numerical result remains provisional until H0 and later theorem/novelty audits pass.

## Strong exact reference

For binary linear codes, the comparison is **not** only against exhaustive codeword scoring. Every trial also runs exact syndrome-trellis ML with work proportional to `n * 2^(n-k)` and checks its metric/decision against exhaustive ML. The reference used in speedup and slope calculations is the cheaper of:

- direct codeword ML, approximately `n * 2^k` likelihood operations;
- exact syndrome-trellis ML, approximately `n * 2^(n-k)` state updates.

This prevents a false high-rate gain.

## Stages

- `E0`: deterministic exactness and theorem-implementation audit;
- `E1`: exact small-DMC representation atlas;
- `E2`: additive and non-cyclic reversible-action positive controls;
- `E3`: non-Latin exact block-scaling tests on binary asymmetric and asymmetric erasure/stuck-at channels;
- `E4`: analytic fiber ceilings and exact/controlled likely-atom confusability graphs.

The driver stops before later stages when a hard earlier gate fails.

## Profiles

- `smoke`: installation and exactness validation only; never use it as the final scientific verdict.
- `standard`: the pre-registered kill-fast investment gate and the launcher default.
- `deep`: confirmation only after `standard` returns `CONTINUE_FIELD_DEFINING_TRACK`, or after a scientifically justified new claim contract. Do not use deep runs to rescue a failed standard claim by silently changing thresholds.

Set the profile before launching:

```bash
export ATOM_GATE_PROFILE=standard
```

## Direct package execution

Inside the extracted package:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
pytest -q
python -m atom_gate.run_gate --profile standard
```

The supplied external launcher performs these steps in `/home/afazeli2006/atom_coding`, stores the virtual environment outside the committed experiment directory, and commits only the package and generated evidence.

## Verdict files

The decisive files are:

- `results/GATE_VERDICT.json` - machine-readable classification and evidence summary;
- `results/GATE_REPORT.md` - concise human-readable report;
- `results/STOP_COMMAND.txt`, `PIVOT_COMMAND.txt`, or `CONTINUE_COMMAND.txt` - exactly one is emitted;
- `results/01_exactness_audit.json`;
- `results/02_small_dmc_atlas_summary.json`;
- `results/04_reversible_action_scaling.json`;
- `results/05_nonlatin_scaling.json`;
- `results/06_rate_survival.json`;
- `results/RESULTS_SHA256.txt`;
- detailed CSV/JSON data and figures used to audit the verdict.

## Interpretation

`CONTINUE_FIELD_DEFINING_TRACK` means only that the programme has survived the early necessary tests. It does not mean the work is field-defining.

`PIVOT_TO_REVISED_TRACK` explicitly stops the original broad claim. No revised path is authorized until a new one-page claim contract and new kill criteria are frozen.

`STOP_BROAD_CHANNEL_ATOM_PROGRAM` means no further investment in the broad proposal is authorized, except verification of the negative result or preparation of a boundary/negative paper.

A finite gate is a research-investment decision, not a universal impossibility proof.
