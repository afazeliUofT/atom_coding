# Proof Audit Checklist - FIBER-GRAND v5

1. **Likelihood identity:** confirm the sum over deletion subsets equals the declared channel likelihood.
2. **Unseen-candidate certificate:** confirm each unseen candidate has at most one unprocessed compatible history per deletion subset and strict inequality excludes unseen ties.
3. **Candidate volume:** prove `2^t B_{n-t}(w) <= V_y(w) <= 2^t binom(n,t) B_{n-t}(w)`.
4. **Exact ambiguity:** prove each complete candidate has exactly `binom(n,t)` inverse histories under the declared inverse representation.
5. **Likelihood sandwich:** verify maximum supporting-history mass lower-bounds aggregate likelihood and the `binom(n,t)` multiple upper-bounds it.
6. **L0 indistinguishability:** verify every unqueried higher-likelihood candidate can be added to a code consistent with the transcript when code cardinality is not exhausted.
7. **Known-cardinality refinement:** verify the necessary alternative: query all higher-likelihood candidates or identify all `M` codewords; check `Q >= min{H_y(c*),M}` and the `h2(q)<R` exponent condition.
8. **Shell offsets:** verify `floor(log_{(1-p)/p} binom(n,t))` and strictness in both lower and upper shells.
9. **Interior-shell exponent:** verify Hamming-ball asymptotics after `O(log n)` shell shifts.
10. **Moment theorem:** verify equality only for history revelation and upper bounds for certified FIBER.
11. **Large deviations:** verify contraction through the binary type rate function.
12. **Phase diagram:** verify the typical boundary uses `h2(p)` and the mean boundary uses `H_{1/2}(p)`.
13. **Parameterized complexity:** confirm the algorithm is XP, not FPT, in edit count.
14. **Numerical exactness:** inspect complete-tie tests and conservative floating-point stopping margin.
