# Z0 Manual Novelty Gate: Primary-Source Boundary

## Claim under audit

The field-defining claim is **not** any of the following:

1. Lempel-Ziv or type-based universal decoding can attain matched random-coding exponents on finite-state channel classes.
2. Randomized guessing of an individual deterministic sequence can be characterized by finite-state compressibility and achieved asymptotically by an LZ-based generator.
3. Deterministic or randomized noise-guessing decoders can be strongly universal over parametric finite-state additive channels using NML, KT, or empirical-entropy metrics.
4. Individual additive-noise communication limits can be expressed through finite-state compressibility in feedback/universal communication models.
5. Guesswork complexity is governed by Renyi-type quantities.

Those foundations already exist in the primary literature.

The only potentially field-defining residue would require a **new theorem-and-construction package** containing all of the following:

- one explicitly frozen, efficiently computable universal codelength with a proved uniform XOR masking law;
- a nontrivial positive-rate explicit code family with a provably favorable description-length spectrum;
- a practical candidate generator and membership architecture whose fully accounted work is competitive at useful blocklengths;
- a theorem not reducible to known randomized individual-sequence guessing or stochastic universal GRAND;
- real post-front-end evidence that the relevant Renyi-1/2 rate lies inside the query-feasible region.

## Primary-source boundary

The audit must include, at minimum:

- J. Ziv and A. Lempel, “Compression of individual sequences via variable-rate coding,” IEEE Transactions on Information Theory, 1978.
- J. Ziv, “Universal decoding for finite-state channels,” IEEE Transactions on Information Theory, 1985.
- A. Lapidoth and J. Ziv, “On the universality of the LZ-based decoding algorithm,” IEEE Transactions on Information Theory, 1998.
- M. Feder and A. Lapidoth, “Universal decoding for channels with memory,” IEEE Transactions on Information Theory, 1998.
- N. Merhav, “Universal decoding for arbitrary channels relative to a given class of decoding metrics,” IEEE Transactions on Information Theory, 2013.
- N. Merhav, “Guessing individual sequences: Generating randomized guesses using finite-state machines,” arXiv:1906.10857, 2019.
- V. Misra and T. Weissman, “The porosity of additive noise sequences,” arXiv:1205.6974, 2012.
- H. K. Miyamoto and S. Yang, “On Universal Decoding over Discrete Additive Channels by Noise Guessing,” arXiv:2501.12971, 2025.
- H. K. Miyamoto, R. Combes, and S. Yang, “Mismatched Exponents for Deterministic and Randomised Noise-Guessing Decoding,” arXiv:2606.26954, 2026.
- Classical CTW, universal prediction, guesswork, universal coding, and computationally bounded-channel literature.

## Mandatory adjudication questions

1. Does unrestricted deterministic tie-breaking make the proposal’s pointwise rank theorem false?
2. After replacing exact rank by a tie-invariant level-set rank or randomized guessing moment, is the remaining theorem already contained in Merhav’s individual-sequence finite-state guessing theory?
3. Is the stochastic finite-state additive core already contained in Miyamoto-Yang and its subsequent mismatched/universal extensions?
4. Is the proposal’s random-coding identity exact for the declared code ensemble, or only exponent-equivalent?
5. Can signed cumulative prediction regret control average block-error probability despite saturation and exponentiation, or is one-sided/exponential-moment control required?
6. Does any explicit LZ78 or CTW codelength satisfy the uniform XOR masking law, including headers and final-phrase conventions?
7. Does description-length code design yield a positive-rate explicit family and a practical finite-block advantage over matched-weight controls?
8. Do real post-front-end impairment traces have a sufficiently small Renyi-1/2 rate after the receiver front end?

## Reference-gate conclusion

The release-standard run gives a negative answer to the field-defining investment question:

- the unrestricted deterministic individual-sequence theorem is structurally false;
- the signed-regret-to-average-reliability implication is structurally false as stated;
- CTW has explicit XOR-masking counterexamples;
- the frozen LZ78 candidate survived finite falsification only and produced no useful short-block code-geometry signal;
- universal finite-state ranking was relatively close to the oracle but violated the proposal’s absolute query budget by many orders of magnitude;
- online adaptation produced no gain over the fixed fitted model in the frozen test;
- no real post-front-end traces were provided to close the systems premise;
- the repaired stochastic/randomized core lies next to, and substantially overlaps, established primary results.

This is not a universal impossibility theorem. It is a preregistered research-investment stop.

## Authoritative action

```text
STOP_FIELD_DEFINING_ZIV_PROGRAMME
DO_NOT_RUN_DEEP_PROFILE
DO_NOT_BEGIN_SOFT_DECODER_OR_HARDWARE_WORK
DO_NOT_BEGIN_A_MULTI_YEAR_CODE_DESIGN_PROGRAMME
```

A short negative/boundary paper may be prepared under a separate publication contract. It must not be represented as continuation of the original four-pillar field-defining programme.
