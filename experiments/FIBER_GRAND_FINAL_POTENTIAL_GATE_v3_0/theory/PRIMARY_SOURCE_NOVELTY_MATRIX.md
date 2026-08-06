# Primary-Source Novelty Adjudication - Fixed-Edit FIBER-GRAND

**Adjudicated scope:** literature searched through 6 August 2026.  This is an evidence-based scientific novelty audit, not a patent opinion or a guarantee concerning unpublished work.

## Overall classification

`NARROW_NOVELTY_SURVIVES`

The original broad claim does not survive.  The narrowed theorem-and-algorithm package was not located as a complete prior result.

| Proposed statement | Closest primary source | What is already known | Residual classification |
|---|---|---|---|
| Guess the most likely additive disturbance and stop at the first codeword | Duffy, Li, Medard, IEEE TIT 2019, arXiv:1802.07010 | GRAND and its membership-only interface | KNOWN |
| Rank arbitrary input candidates by channel likelihood and query code membership | Tan and Joudeh, IEEE TIT 2025, arXiv:2502.05959 | General DMC guessing-based decoding with abandonment and asymptotics | KNOWN |
| Unified noise-side and codeword-side guessing | Wang et al., arXiv:2511.12108 | GRAND/GCD exact stopping and operating regimes | KNOWN |
| Best path may differ from best aggregate string | de la Higuera and Oncina, J. Logic Comput., DOI 10.1093/logcom/exs049; weighted-automata literature | Best-string versus best-path distinction and hardness | KNOWN |
| Priority-first exact ML search | Han, Hartmann, Chen, IEEE TIT 1993, DOI 10.1109/18.259636 | A*/priority-first code-trellis ML decoding | KNOWN |
| Sequential decoding with synchronization errors | Gallager, Lincoln Lab Group Report 2502, 1961; Banerjee, Lenz, Wachter-Zeh, arXiv:2201.11935 | Drift-state sequential decoding and computation analysis for convolutional codes | KNOWN |
| Sum hidden alignment paths | Davey and MacKay, IEEE TIT 2001; Mohri, weighted automata algorithms | Forward-backward/transducer probability summation | KNOWN |
| Enumerate insertion/deletion frames, decode, score, select | US4922494A (1990) | Error-frame enumeration and probability-based word selection | PARTIALLY ANTICIPATED |
| GRAND-like repair of demapper-induced insertions/deletions | Ozaydin, Medard, Duffy, arXiv:2210.16187; later GRAND-assisted demodulation | Specialized lightweight GRAND in variable-length modulation | PARTIALLY ANTICIPATED |
| Embedding/alignment multiplicity in fixed-k deletion likelihood | Sabary et al., arXiv:2201.02466 and related deletion-ML work | Multiplicity-aware likelihood and special one/two deletion decoders | KNOWN |
| Codebook-free membership-first aggregate inverse generation for a fixed edit budget, with a strict sum-of-frontiers certificate | No primary source located containing the full construction and claim boundary | Ingredients exist separately | DISTINCT, NARROW |
| Deterministic shell-certificate work bound for exactly t uniform deletions plus survivor BSC noise | No identical theorem located | Type-volume and sequential-decoding techniques are classical | DISTINCT, POSSIBLY ELEMENTARY |
| Matching moment exponent for probability-ordered fixed-edit history revelation and upper moment/tail law for certified FIBER | No identical fixed-edit result located | Arikan guesswork and conditional guesswork supply the Renyi calculus | DISTINCT, NARROW |
| Typical/mean phase diagram against a redundancy-state exact decoder | No identical fixed-edit theorem located | GRAND/GCD regime comparisons and general DMC abandonment exponents are prior art | DISTINCT SYNTHESIS |

## Claim authorized after this audit

> For a fixed number of uniformly located deletions followed by survivor BSC substitutions, a membership-only aggregate inverse decoder can provide a strict exact-ML certificate.  Probability-ordered fixed-edit history revelation has the same moment exponent as BSC guesswork up to subexponential edit overhead, while certified FIBER has the corresponding upper moment and tail bounds.  A hybrid with a redundancy-state exact search yields distinct typical and mean complexity phase boundaries.

## Claims prohibited

- `fiber guesswork` as a new information-theoretic random variable;
- novelty of path summation, A*, branch-and-bound, or generic candidate likelihood ordering;
- polynomial or FPT exact decoding for unrestricted edit rates or weighted transducers;
- practical superiority over synchronization coding without matched system tests.
