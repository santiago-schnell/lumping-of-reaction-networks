# Singular computations

Computer-algebra scripts for the ideal decompositions reported in Section 6.
Each script is plain text and reproducible from the command line with
[Singular](https://www.singular.uni-kl.de/) (tested against the 4-3-x series;
`primdec.lib` ships with the distribution).

| File | Paper reference | What it computes |
| --- | --- | --- |
| `two_pathway_enzyme_decomposition.sing` | Section 6.2 / Appendix | `facstd` decomposition of the critical-parameter ideal (the paper reports 92 components), then elimination of the lumping parameters `t1..t9` and `minAssChar` of the elimination ideal. |
| `replication_decomposition.sing` | Section 6.1 / Appendix | `minAssChar` of the critical-parameter ideal in `C[p,q,r,k_i]` (the paper reports 22 irreducible components). |

Run, for example:

```bash
Singular singular/two_pathway_enzyme_decomposition.sing
Singular singular/replication_decomposition.sing
```

## Provenance of the ideals

- The enzyme ideal is the raw output of
  `mathematica/two_pathway_enzyme_critical_variety.wl`; it is transcribed here
  verbatim so the `facstd` decomposition can be reproduced exactly.
- The replication generators are the nonzero coefficient conditions of the
  critical-parameter system, i.e. the nonzero rows of the matrix in
  Section 6.1.  They are produced by
  `mathematica/replication_critical_variety.wl` and reproduced in Python by
  `examples/example_section_6_gpl_replication.py`.

The `facstd` step for the enzyme ideal takes a few minutes; the published
output is sensitive to the monomial ordering (`dp`) and to `option(redSB)`,
both set at the top of the script.
