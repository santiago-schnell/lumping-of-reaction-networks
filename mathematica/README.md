# Mathematica / Wolfram Language workflow

This folder contains the symbolic computation that produces the critical-parameter
ideal for the two-pathway enzyme example in Section 6.2 of the paper.

## Files

| File | Description |
|------|-------------|
| `two_pathway_enzyme_critical_variety.wl` | Plain-text Wolfram Language script. Reproducible from the command line; the right entry point for review and version control. |
| `lump_6dim_ex50_d3_B1.nb` | Original Mathematica notebook (Mathematica 13+). Re-run from a clean kernel after applying the same correction. |

## What the script does

For the two-pathway enzyme system

```
S + E  <->  C1        (k1, km1)
C1     <->  C         (k2, km2)
S + E  <->  C2        (k3, km3)
C2     <->  C         (k4, km4)
C      <->  E + P     (k5, km5)        <-- reversible
```

the script:

1. Defines the species order `x1=s, x2=e, x3=c1, x4=c2, x5=c, x6=p`.
2. Sets up the row-echelon lumping ansatz `T(t1,...,t9)` and the right-kernel
   basis `B(t1,...,t9)`, and verifies `T . B == 0`.
3. Defines the **corrected** mass-action vector field. The key correction over
   the prior notebook is the inclusion of `+ km5 * x2 * x6` in `x5d`, which is
   required so that
   - `x2 + x3 + x4 + x5` (total enzyme), and
   - `x1 + x3 + x4 + x5 + x6` (substrate--product balance)
   are conserved.  The script prints an error message if either conservation
   law fails.
4. Computes the Jacobian and `U = T . jac . B`.
5. Extracts the coefficients of `U` in the state variables `x1..x6`, which gives
   the polynomial conditions in the rate constants and lumping parameters.
6. Splits the conditions into their constant and rate-linear parts using
   `CoefficientArrays`, and verifies that the constant part is zero
   (i.e.\ the system is homogeneous-linear in the rate constants for fixed `T,B`).
7. Highlights the `km5`-sensitive conditions, which are the ones that change
   under the correction.

## How to run

From this folder, either:

```bash
wolframscript -file two_pathway_enzyme_critical_variety.wl
```

or, in a Mathematica kernel:

```
<< two_pathway_enzyme_critical_variety.wl
```

The script prints diagnostic output (kernel check, conservation checks, number
of coefficient conditions, the `km5`-sensitive conditions, and the dimensions
of the coefficient matrix `M`).  No file output is written.

## Exporting the ideal to Singular

The list `conditions` (after running the script) is the generator set for the
critical ideal.  To export to Singular syntax, run e.g.

```mathematica
StringJoin[Riffle[ToString /@ conditions, ",\n"]]
```

and paste the result into a Singular `ideal g = ...;` declaration.

## Reproducibility notes

* Avoid using Mathematica's previous-output symbol `%`.  This script uses named
  variables throughout so it is robust to being re-run cell by cell.
* The first line is `ClearAll["Global`*"]`, which removes any stale definitions
  before the script begins.
* Unused species variables `x7`, `x8` from earlier drafts have been removed.
