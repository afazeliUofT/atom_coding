# Release Validation - FIBER-GRAND Pre-Manuscript Closure v5.0

## Purpose

Version 5 is a mandatory closure gate after commit `2f67a32`. It does not assume that the version-4 performance classification is final. It tests the two unresolved paper-readiness issues: feasible syndrome-trellis coverage and supported positive-error/tail evidence.

## Validation completed

1. Python package imports and all 21 tests pass.
2. The C++20 benchmark compiles using `-O3 -DNDEBUG -std=c++20 -march=native -Wall -Wextra`.
3. The compiled self-test returns `SELF_TEST_PASS cases=600`.
4. Direct one-block probes confirm that the syndrome-trellis decoder is available, certified, and tie-consistent at:
   - `n=48, R=0.75` (`18,874,368` estimated updates);
   - `n=48, R=0.875` (`294,912` estimated updates);
   - `n=64, R=0.875` (`2,097,152` estimated updates).
5. The smoke profile completes all scientific stages. Its underpowered `NARROW` result is a software-validation outcome only and is not an investment verdict.
6. The theory PDF is searchable, openable, preflighted, and visually inspected on representative pages.
7. The package excludes generated virtual environments, build binaries, caches, and prior result files.

## Standard profile

The authoritative standard profile deliberately requires 500+ largest-block trials per key configuration, three timing repeats, positive-error strata, and feasible trellis coverage. It is intended to run in the user's WSL environment and was not precomputed during package construction. Its result must be adjudicated from the newly generated `GATE_VERDICT.json`.

## Claim discipline

The release does not restore generic fiber-guesswork, generic A*/path-summation novelty, unrestricted transducer tractability, FPT edit dependence, processor-cycle optimality, real-system relevance, or field-defining status.
