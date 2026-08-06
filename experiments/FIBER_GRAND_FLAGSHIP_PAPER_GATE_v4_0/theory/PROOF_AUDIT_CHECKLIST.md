# Proof Audit Checklist

1. **Likelihood identity** - verify one compatible inverse history per deletion set and candidate.
2. **Strict certificate** - verify that the sum of next stream-head masses bounds every completely unseen candidate and that strict inequality gives the complete ML tie set.
3. **Candidate volume** - verify injection for a fixed deletion set and the union upper bound.
4. **Exact ambiguity degree** - verify exactly `binom(n,t)` complete histories per candidate.
5. **L0 oracle lower bound** - verify the arbitrary-code oracle indistinguishability model and the need to query every higher-likelihood candidate.
6. **Shell offset** - check `L=floor(log_{(1-p)/p} binom(n,t))` and strictness of `a^(L+1)>S`.
7. **FIBER upper query bound** - verify discovery by shell `d*` and certification by shell `d*+L`.
8. **Interior-shell asymptotics** - verify uniform Hamming-ball estimates under `d*/(n-t)->q in (0,1/2)` and polynomial ratio.
9. **Moment equality boundary** - equality only for history revelation; certified decoder gets an upper bound.
10. **Large deviations** - verify contraction and endpoint at `gamma=1`.
11. **XP boundary** - do not call `binom(n,t)` FPT.
12. **Numerics** - verify complete ties and conservative floating-point comparisons.
