# Lumping of reaction networks (Python)

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A Python package for analyzing **exact linear lumping** of parameter-dependent **mass action** reaction networks (and closely related polynomial ODE models).

This repository accompanies the paper draft:

> **“Lumping of reaction networks: Generic and critical parameters”** (draft dated **February 4, 2026**)

The software implements the main computational ingredients from the paper:

1. **Generic (parameter-independent) lumpings** (paper Section 3):
   - **Type 1:** eliminate *common non-reactant species*;
   - **Type 2:** project along *stoichiometric first integrals*.
2. **Critical parameters** for a prescribed lumping map **T** (paper Section 4):
   algebraic conditions on the rate constants under which **T** becomes an exact linear lumping.
3. **Proper lumpings** (paper Section 5):
   partition-based lumping maps with a dedicated Jacobian block / column-sum criterion.

In addition to these core computations, the package provides small “quality of life” utilities:

- a reaction-network string parser (`"A + B <-> C"`),
- compact reporting helpers for candidate reductions,
- optional numerical validation for small symbolic problems.

---

## Contents

- [Installation](#installation)
- [If you are new to Python](#if-you-are-new-to-python)
- [Repository layout](#repository-layout)
- [Quick start](#quick-start)
- [Core workflows](#core-workflows)
  - [1) Define a network](#1-define-a-network)
  - [2) Compute generic lumpings](#2-compute-generic-lumpings)
  - [3) Critical parameters for a prescribed T](#3-critical-parameters-for-a-prescribed-t)
  - [4) Constrained lumping](#4-constrained-lumping)
  - [5) Proper lumping (partitions)](#5-proper-lumping-partitions)
  - [6) Enumerate candidate reductions and print a report](#6-enumerate-candidate-reductions-and-print-a-report)
- [Computer algebra workflows](#computer-algebra-workflows)
  - [Row-echelon ansatz enumeration (general linear lumpings)](#row-echelon-ansatz-enumeration-general-linear-lumpings)
  - [Component enumeration and decomposition](#component-enumeration-and-decomposition)
- [Numerical validation](#numerical-validation)
- [Troubleshooting](#troubleshooting)
- [License](#license)
- [How to cite](#how-to-cite)

---

## Installation

### Requirements

- Python **≥ 3.8**
- SymPy **≥ 1.12**
- NumPy **≥ 1.20**
- SciPy **≥ 1.7**

### Install (recommended: editable install)

From the repository root:

```bash
pip install -e .
```

That installs the `lumping_analysis` package so you can import it from anywhere.

---

## If you are new to Python

If you do not typically work in Python, this is the minimum setup that tends to work well.

### 1) Create and activate a virtual environment

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows (PowerShell)**

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
```

### 2) Install the package

```bash
pip install -e .
```

### 3) Run an example script

```bash
python examples/example_4_2_three_species.py
python examples/example_constrained_michaelis_menten.py
```

If you see printed conditions and/or a reduced model description, your environment is working.

---

## Repository layout

- `src/lumping_analysis/`
  - `network.py`, `reaction.py`, `parser.py`: define reaction networks and mass-action ODEs
  - `analyzer.py`: generic lumping, critical-parameter computations, proper lumping
  - `report.py`: compact console/Markdown reporting utilities
- `examples/`: runnable scripts aligned with paper examples and common workflows
- `tests/`: a small test suite (useful as executable documentation)
- `basic_usage.py`: a minimal one-file usage demo

### Example scripts

The `examples/` directory contains runnable scripts that mirror common use-cases
and (where applicable) align with examples in the paper draft.

- `examples/example_4_2_three_species.py`: three-species linear network (paper Example 4.2)
- `examples/example_constrained_michaelis_menten.py`: constrained reduction for reversible Michaelis–Menten (paper Example 4.4)
- `examples/example_proper_lumping_michaelis_menten.py`: proper lumping criterion demonstration (paper Section 5)
- `examples/search_proper_lumpings_michaelis_menten.py`: enumerate proper lumpings (small search)
- `examples/example_section_6_gpl_replication.py`: self-replication case study setup (paper Section 6)
- `examples/enumerate_reductions_michaelis_menten.py`: batch enumeration + report printing

You can run any of these from the repository root via `python examples/<script>.py`.

---

## Quick start

```python
import sympy as sp
from lumping_analysis import michaelis_menten_network, LumpingAnalyzer

# Built-in network (reversible Michaelis–Menten)
net = michaelis_menten_network()
print(net.summary())

an = LumpingAnalyzer(net)

# Generic lumpings (Type 1 and Type 2)
generic = an.find_generic_lumpings()
print("Common non-reactants:", generic["common_non_reactants"])
print("Stoichiometric first integrals:")
for mu in generic["stoichiometric_integrals"]:
    print("  ", mu)

# Critical parameters for a prescribed candidate lumping matrix T
# (variables ordered as [S, E, C, P] in the built-in model)
T = sp.Matrix([
    [1, 0, 1, 1],  # S + C + P
    [0, 1, 1, 0],  # E + C
    [0, 0, 1, 0],  # C
])

res = an.find_critical_parameters(T)
print("Critical conditions:")
for c in res["conditions"]:
    print("  ", sp.factor(c), "= 0")
```

---

## Core workflows

### 1) Define a network

#### A. From reaction strings (recommended)

```python
from lumping_analysis import ReactionNetwork

net = ReactionNetwork.from_string("A + B <-> C")

# Multiple reactions (semicolon or newline separated)
net = ReactionNetwork.from_string(
    """
    S + E <-> C
    C <-> E + P
    """,
    species_names=["S", "E", "C", "P"],
)
```

#### B. Programmatically (full control)

```python
import sympy as sp
from lumping_analysis import Reaction, ReactionNetwork

k1, km1 = sp.symbols("k1 km1", positive=True)

# Species ordering: [A, B, C]
reactions = [
    Reaction((1, 1, 0), (0, 0, 1), k1),   # A + B -> C
    Reaction((0, 0, 1), (1, 1, 0), km1),  # C -> A + B
]

net = ReactionNetwork(n_species=3, reactions=reactions, species_names=["A", "B", "C"])
```

---

### 2) Compute generic lumpings

For **generic parameters** (paper Section 3), exact linear lumpings are restricted to the “obvious” families:

- eliminate *common non-reactants* (Type 1), and/or
- use *stoichiometric first integrals* (Type 2).

```python
generic = an.find_generic_lumpings()

print("Common non-reactant species:", generic["common_non_reactants"])
print("Stoichiometric first integrals:")
for mu in generic["stoichiometric_integrals"]:
    print(mu)
```

The function returns both human-readable species names (when available) and indices.

---

### 3) Critical parameters for a prescribed T

The critical-parameter test implemented here is the Jacobian invariance condition:

- Let **T** be an `e × n` full-row-rank matrix and let columns of **B** be a basis of `ker(T)`.
- Then **T** is a lumping map for the specialization `x' = F(x,k*)` iff

```text
T · DF(x,k*) · B = 0    for all x.
```

Computationally:

1. build the symbolic Jacobian `J = DF(x,k)`,
2. compute a symbolic kernel basis `B` for `ker(T)`,
3. form `TJB = T*J*B`,
4. extract **polynomial coefficient conditions** in the state variables `x`.

```python
import sympy as sp

T = sp.Matrix([
    [1, 0, 1, 1],
    [0, 1, 1, 0],
])

res = an.find_critical_parameters(T)
print("Conditions:")
for c in res["conditions"]:
    print("  ", sp.factor(c), "= 0")

# If the conditions are linear in the rate constants, the code also returns
# an rref-based analysis of the induced linear system.
analysis = res["solutions"].get("linear_system_analysis")
if analysis:
    print("Linear relations among rate constants:")
    for r in analysis["relations"]:
        print("  ", r, "= 0")
```

---

### 4) Constrained lumping

In applications, you often want the reduced coordinates to include specific observables (e.g. conservation laws). The package supports building a “constrained” ansatz:

- you prescribe some rows of **T**,
- you add a chosen number of additional free rows,
- then you compute critical conditions for the resulting symbolic **T**.

```python
integrals = net.stoichiometric_first_integrals()

T, free_params = an.build_constrained_lumping_matrix(
    prescribed_rows=integrals,
    n_free_rows=1,
    free_param_prefix="p",
)

res = an.find_critical_parameters(T)
print("Free parameters:", free_params)
print("Conditions:")
for c in res["conditions"]:
    print(c, "= 0")
```

---

### 5) Proper lumping (partitions)

A **proper lumping** groups species into blocks and lumps by block-sums

```text
y_p = Σ_{j∈I_p} x_j
```

The paper (Section 5) provides an efficient **Jacobian block / column-sum criterion** for proper lumpings; the package implements it directly.

```python
import sympy as sp

res = an.proper_lumping_conditions([["S", "P"], ["E"], ["C"]])
print("T =")
sp.pprint(res["T"])
print("Conditions:")
for c in res["conditions"]:
    print("  ", sp.factor(c), "= 0")
```

You can also enumerate partitions for small networks:

```python
results = an.find_proper_lumpings(n_blocks=3, max_partitions=10_000)
for r in results:
    if r["conditions"]:
        print(r["blocks"], r["conditions"])
```

---

### 6) Enumerate candidate reductions and print a report

For exploratory work, it is often convenient to enumerate a collection of *structured* candidate reductions (generic families, proper lumpings, constrained ansätze) and print them in a compact way.

```python
from lumping_analysis import (
    LumpingAnalyzer,
    ReductionReportOptions,
    format_reduction_report,
    michaelis_menten_network,
)

net = michaelis_menten_network()
an = LumpingAnalyzer(net)

results = an.enumerate_reductions(
    include_generic=True,
    include_proper=True,
    proper_n_blocks=3,
    max_partitions=10_000,
    # Optional: discard candidates that force any of these rates to be zero.
    required_nonzero_rates=net.rate_constants,
)

print(
    format_reduction_report(
        net,
        results,
        options=ReductionReportOptions(max_conditions=12, max_relations=12),
    )
)
```

This does **not** attempt to enumerate *all* linear lumpings (which is combinatorially large); see the [Computer algebra workflows](#computer-algebra-workflows) section for the general strategy.

---

## Computer algebra workflows

The paper emphasizes that the *theory* reduces many questions to solving finitely many polynomial systems, but the practical feasibility depends on the size of the network and the algebraic-geometry tooling used.

The Python code in this repository intentionally stays “lightweight” (SymPy-only). For serious ideal decomposition / component enumeration, you will typically want to use an open-source CAS that is designed for that purpose.

### Row-echelon ansatz enumeration (general linear lumpings)

Paper Section 4 (see the row-echelon discussion leading to Proposition 5) gives a canonical way to represent general full-row-rank lumping matrices up to row operations.

Fix a target reduced dimension `e < n`.

1. Choose a set of **pivot columns** `P ⊂ {1,…,n}` with `|P| = e`.
2. Permute columns so the pivots come first.
3. Put **T** into (reduced) row-echelon form

```text
T = [ I_e  |  T̂ ]
```

where the entries of `T̂` are free symbolic parameters.

4. A convenient kernel basis is then

```text
B = [  T̂  ]
    [ -I_{n-e} ]
```

5. Insert `T` and `B` into the critical condition

```text
T · DF(x,k) · B = 0  (for all x)
```

and extract coefficient conditions in `x`. This yields a polynomial system in:

- the rate constants `k` (or a selected subset of parameters), and
- the ansatz parameters (the entries of `T̂`).

This is the “row-echelon ansatz enumeration” approach:

- Enumerate pivot sets `P`.
- For each `P`, set up the corresponding ansatz system.
- Solve / decompose each system (or filter it) to obtain candidate lumpings and corresponding critical parameter relations.

**Practical tips.**

- Start with small `e` (e.g. `e = n-1` or `e = n-2`) and/or constrain the form of `T̂`.
- Use chemical constraints to prescribe some rows (constrained lumping) and reduce the number of free ansatz parameters.
- Use quick symbolic rank tests / rref diagnostics to discard cases that immediately force unwanted rates to be zero.

### Component enumeration and decomposition

Once you have a polynomial condition list `c_1, …, c_m` (in parameters and ansatz variables), you can form the ideal

```text
I = ⟨ c_1, …, c_m ⟩
```

and then decompose the corresponding variety into components.

Typical tasks include:

- computing minimal associated primes (component enumeration),
- primary decomposition, or
- elimination (e.g. eliminate ansatz variables to obtain conditions purely in the rate constants).

**Open-source CAS options.**

- **Singular** (specialized for polynomial ideals; used in many algebraic-geometry workflows)
- **SageMath** (can call Singular and other backends)
- **Macaulay2** (specialized for commutative algebra)

SymPy can compute Gröbner bases but does not aim to compete with specialized systems for large decompositions.

**A minimal Singular workflow (practical template).**

Below is a small *template* you can adapt (Singular syntax). The key ideas are:

- declare **all ansatz variables + parameters** as ring variables,
- define the ideal generated by the critical conditions,
- optionally compute a decomposition (component enumeration), and
- optionally **eliminate** ansatz variables to obtain parameter-only relations.

```singular
// Example template (edit variables + conditions)
ring R = 0,(t1,t2,t3,k1,km1,k2,km2),dp; // dp = degree reverse lex (common choice)
option(redSB);

ideal I =
  t1 + t2 - 1,
  k1 - km2,
  t3*(k2 - km1);

// 1) Gröbner basis (often useful even before decomposition)
ideal G = groebner(I);

// 2) Component enumeration / decomposition (choose one)
// list L = minAssChar(I);   // minimal associated primes
// list L = facstd(I);       // factorized standard basis components

// 3) Elimination (example: eliminate ansatz variables t1,t2,t3)
ideal I_param = eliminate(I, t1*t2*t3);
I_param;
```

How you *order variables* matters for elimination; many users place the variables
to eliminate earlier in the order, or switch to an elimination order.

If you prefer a single toolchain, SageMath can call Singular internally, so you
can keep a Python-based workflow while delegating heavy algebra to Singular.

When interpreting components, remember that reaction-network rate constants are typically constrained by:

- nonnegativity (and often strict positivity for reactions assumed present),
- application-specific “admissible parameter” restrictions.

So it is common that many algebraic components correspond to degenerate networks where some rates are forced to zero; those can be filtered out depending on your modeling intent.

---

## Numerical validation

For a *small* exact lumping, you can optionally attempt to construct the reduced polynomial system `y' = G(y)` (Li–Rabitz viewpoint) and compare full vs reduced simulations.

```python
import numpy as np
import sympy as sp
from lumping_analysis import michaelis_menten_network, LumpingAnalyzer

net = michaelis_menten_network()
an = LumpingAnalyzer(net)

T = sp.Matrix([
    [1, 0, 1, 1],
    [0, 1, 1, 0],
])

k1, km1, k2, km2 = net.rate_constants
kvals = {k1: 1.0, km1: 0.5, k2: 0.8, km2: 1.0}

x0 = np.array([1.0, 0.5, 0.0, 0.0])
val = an.validate_numerically(T, kvals, x0, t_span=(0.0, 10.0))
print(val)
```

For larger systems, the symbolic construction of `G` can be expensive; in that case you can still use the **condition error** reported by `validate_numerically` as a diagnostic.

---

## Troubleshooting

### “No solutions found” (or the conditions look inconsistent)

- Factor the returned conditions (`sympy.factor`) and check for obvious contradictions.
- If the resulting polynomial system is large, consider exporting it to a dedicated CAS (see above).

### “ker(T) is trivial”

Your **T** has full rank `n` (no reduction). Reduce the number of rows of **T**.

### Symbolic computations are slow

- Try smaller examples first (the built-in examples are chosen to be manageable).
- Use constrained ansätze to reduce the number of free symbolic parameters.
- Use partition-based proper lumping for quick screening.

---

## License

MIT License. See `LICENSE`.

---

## How to cite

If you use this code in research, please cite the accompanying paper draft.
