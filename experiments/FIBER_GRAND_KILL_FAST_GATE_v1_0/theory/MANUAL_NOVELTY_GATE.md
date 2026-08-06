# F0 Manual Novelty Gate: FIBER-GRAND

## Claim under audit

The field-defining claim is **not** any of the following individually:

1. A non-invertible channel likelihood sums probabilities over latent alignments or paths.
2. Maximum-likelihood sequence decoding can be performed with dynamic programming, weighted finite-state transducers, sequential decoding, or priority-first/A* search.
3. A branch-and-bound decoder may stop when an incumbent lower bound exceeds every frontier upper bound.
4. The true input can be ranked by its posterior probability given the output.
5. Deletion/insertion likelihood depends on embedding or alignment multiplicity.

Those foundations have substantial prior art.

The potentially new residue is the complete operational package:

> A channel-side, code-modular aggregate inverse-search engine for non-invertible random transformations that uses only a generic code interface for correctness, scores each candidate by all compatible histories, supplies an auditable exact-ML certificate, materially reduces fully accounted finite-block work, and supports a new ambiguity/search-inflation or tractability theorem.

## Mandatory primary-source boundary

An independent audit must include at least:

- Duffy, Li, and Médard, GRAND (2019).
- Arıkan, guessing with side information (1996), and later nonasymptotic conditional-guesswork work.
- Gallager's sequential decoding with synchronization errors (1961).
- Han, Hartmann, and Chen's priority-first ML block decoding (1993).
- Davey-MacKay and other forward/backward synchronization decoders.
- Weighted finite-state transducer algorithms and determinization/ambiguity literature.
- Most-probable-string hardness for ambiguous probabilistic automata.
- ML deletion/insertion decoding and embedding-multiplicity literature.
- VT, Guess-and-Check/GC+, marker/watermark, synchronization-string, and polar insertion/deletion work.

## Mandatory adjudication questions

1. Is the exact membership-first history decoder already present under another synchronization-decoding name?
2. Is the multi-stream unseen-candidate bound a known A*/branch-and-bound heuristic in equivalent form?
3. Is the prefix implementation simply a standard priority-first decoder on a combined channel/code trellis?
4. Is any proposed "fiber guesswork" theorem no more than classical posterior conditional guesswork?
5. What theorem about search inflation, ambiguity, fixed-edit complexity, or resource lower bounds is genuinely new?
6. Does the method beat strong exact and specialized baselines after all costs are charged?
7. Does code modularity survive without relying on a code-specific completion oracle?
8. Is there a calibrated physical synchronization impairment where the model and latency regime are credible?

## Release status

**PENDING INDEPENDENT PRIMARY-SOURCE ADJUDICATION.**

The standard computational gate may authorize a revised research path, but it cannot close this novelty gate.

## Failure action

If the surviving synthesis is already known or no new theorem remains after removing standard conditional guesswork and branch-and-bound results, emit:

```text
STOP_FIELD_DEFINING_FIBER_PROGRAMME
```

A narrower algorithm, implementation, or application paper may be evaluated under a new claim contract.
