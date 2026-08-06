# Independent proof-audit checklist

An independent reviewer should verify, without relying on the implementation:

1. the unseen-candidate residual sum bound and strict tie-complete stopping condition;
2. the deterministic shell-certificate index, including `p=0`, shell saturation, and the factor `2^t binom(n,t)`;
3. the reveal-rank sandwich under arbitrary tie order within a shell;
4. the method-of-types/Laplace proof of the moment exponent `rho H_{1/(1+rho)}(p)`;
5. the contraction-principle proof of the reveal-work large-deviation rate;
6. the distinction between an equality for history revelation and only an upper bound for actual certified decoder work;
7. the hybrid phase diagram and every oracle/preprocessing cost assumption;
8. the XP, rather than FPT, dependence on edit count;
9. numerical-certification assumptions and complete ML tie handling.

Any defect that removes the moment/tail theorem or exact certificate triggers `STOP_FIBER_FIELD_DEFINING_PROGRAMME`.
