# Manuscript figures

This directory contains the source files and generated PDFs for the figures used in the manuscript.

## Files

| Figure | Source | Generated PDF |
| --- | --- | --- |
| Type 1 / Type 2 invariant subspaces | `type1_type2_figure.py` | `generated/type1_type2_subspaces.pdf` |
| Generic-lumping algorithm flowchart | `algorithm_flowchart_v7.tex` | `generated/algorithm_flowchart.pdf` |
| Reversible Michaelis--Menten network | `michaelis_menten_v2.py` | `generated/michaelis_menten_network.pdf` |
| Proper-lumping block structure | `proper_lumping_figure.py` | `generated/proper_lumping_blocks.pdf` |

## Rebuild

From the repository root:

```bash
python figures/type1_type2_figure.py
python figures/michaelis_menten_v2.py
python figures/proper_lumping_figure.py
```

The Python scripts write PDFs and PNGs to `figures/generated/`.

To rebuild the TikZ flowchart, use a LaTeX installation with the `standalone` class, for example:

```bash
cd figures
pdflatex algorithm_flowchart_v7.tex
mv algorithm_flowchart_v7.pdf generated/algorithm_flowchart.pdf
```
