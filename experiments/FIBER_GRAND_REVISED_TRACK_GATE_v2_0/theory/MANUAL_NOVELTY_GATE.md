# Final Manual Novelty Gate for the Revised FIBER-GRAND Track

## Claim under adjudication

The claim is **not** that hidden paths must be summed, that a best path can differ from a best string, that priority-first decoding exists, or that posterior rank is a new random variable. Those ideas are established.

The surviving candidate claim is the following theorem-and-algorithm synthesis:

> For bounded synchronization edits, a membership-only aggregate inverse decoder can provide exact ML certificates and, for a fixed number of uniform deletions plus BSC substitutions, its history work has the BSC guessing exponent up to fixed-edit polynomial overhead, while remaining competitive with code-specific exact search.

## Required primary-source comparisons

1. GRAND and its general invertible-operation formulation.
2. Gallager's sequential decoding for synchronization errors.
3. Han--Hartmann--Chen priority-first ML decoding.
4. Davey--MacKay marker/watermark forward--backward decoding.
5. Weighted automata and probabilistic transducer inference.
6. Most-probable-string / best-string complexity for ambiguous probabilistic automata.
7. VT, shifted-VT, Guess-and-Check/GC+, synchronization strings, and polar IDS constructions.
8. ML/ML* results for fixed-k deletion channels.
9. Conditional guesswork and Arimoto-Renyi side-information theory.
10. Contemporary patents, preprints, conference papers, and code repositories using GRAND-like edit enumeration.

## Mandatory questions

- Has the deterministic shell-certificate theorem already appeared, perhaps under sequential decoding or bounded-edit search terminology?
- Is the claimed `h2(p)` fixed-edit history exponent an immediate corollary of an established theorem, or does the aggregate certificate add a genuinely new result?
- Has a code-modular membership-only exact ML decoder for one deletion plus substitutions already been formulated and benchmarked?
- Is membership-first complete scoring new or a standard branch-and-bound simplification?
- Are the search-inflation and history-duplication quantities operationally new, or renamed standard search overheads?
- Does the revised algorithm add anything beyond an implementation of known A*/trellis principles?

## Decision

- If the theorem and complete synthesis are already established: `STOP_FIELD_DEFINING_NOVELTY_CLAIM`.
- If only the one-deletion implementation is new: narrow to an algorithm paper.
- If the theorem, code-modular architecture, and strong-baseline evidence survive: proceed to calibrated physical validation.

A positive computational result is never accepted as a novelty proof.
