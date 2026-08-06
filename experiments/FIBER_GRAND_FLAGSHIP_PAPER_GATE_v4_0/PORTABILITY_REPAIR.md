# FIBER-GRAND Flagship Gate v4.0 R1 Portability Repair

The original v4.0 package passed its Python theory and exactness stages in the user environment but failed to compile on GCC/libstdc++ because `std::powl` and `std::fabsl` were not provided in namespace `std`.

R1 makes only these source-level substitutions:

- `std::powl(a,b)` -> `std::pow(a,b)`
- `std::fabsl(x)` -> `std::fabs(x)`
- suppresses the unused deletion-length parameter warning without changing behavior.

C++ overload resolution retains `long double` arithmetic. No theorem, algorithm, benchmark configuration, seed, threshold, or scientific decision rule was changed.
