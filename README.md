# Lumping Analysis for Mass Action Reaction Networks

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A Python module for analyzing linear lumping of parameter-dependent mass action reaction networks using computer algebra.

## Paper

This software accompanies the paper to be submitted:

> **"Lumping of reaction networks: Generic and critical parameters"**  
> J.E., V.R., S.S., S.W.  
> *SIAM Journal on Applied Dynamical Systems* (2025)

If you use this software in your research, please cite the paper.

## Overview

Linear lumping is a dimension reduction technique for chemical reaction networks. Given an ODE system

$$\dot{x} = F(x, k), \quad x \in \mathbb{R}^n, \quad k \in \mathbb{R}^d$$

a linear map $T: \mathbb{R}^n \to \mathbb{R}^e$ (with $e < n$) defines a **lumping** if the reduced variables $y = Tx$ satisfy a closed system $\dot{y} = G(y, k)$.

This module provides tools to:

1. **Define reaction networks** symbolically with mass action kinetics
2. **Find generic lumpings** (Type 1 and Type 2) valid for all parameters
3. **Identify critical parameters** where non-trivial lumpings become available
4. **Validate lumpings numerically** with error quantification
5. **Export equations to LaTeX** for publication

## Installation

### Requirements

- Python ≥ 3.8
- SymPy ≥ 1.12
- NumPy ≥ 1.20
- SciPy ≥ 1.7

### Install Dependencies

```bash
pip install sympy numpy scipy
```

### Install the Module

Simply copy `lumping_analysis.py` to your working directory or add it to your Python path:

```bash
# Option 1: Copy to your project
cp lumping_analysis.py /path/to/your/project/

# Option 2: Install as editable package (if setup.py is provided)
pip install -e .
```

## Quick Start

```python
from lumping_analysis import michaelis_menten_network, LumpingAnalyzer
import sympy as sp

# Load a built-in network
network = michaelis_menten_network()
print(network.summary())

# Create analyzer
analyzer = LumpingAnalyzer(network)

# Find generic (parameter-independent) lumpings
generic = analyzer.find_generic_lumpings()
print(f"Stoichiometric first integrals: {len(generic['stoichiometric_integrals'])}")

# Define a candidate lumping matrix and find critical parameters
T = sp.Matrix([
    [1, 0, 1, 1],  # y₁ = S + C + P
    [0, 1, 1, 0],  # y₂ = E + C (enzyme conservation)
])
result = analyzer.find_critical_parameters(T)
print("Critical conditions:", result['conditions'])
```

## Defining Reaction Networks

### Method 1: String Parser (Recommended)

The easiest way to define networks using chemical notation:

```python
from lumping_analysis import ReactionNetwork

# Simple reversible reaction
network = ReactionNetwork.from_string("A + B <-> C")

# Multiple reactions (semicolon or newline separated)
network = ReactionNetwork.from_string("""
    S + E <-> C
    C -> E + P
""")

# With stoichiometric coefficients
network = ReactionNetwork.from_string("2A + B -> C")

# Specify species ordering
network = ReactionNetwork.from_string(
    "S + E <-> C; C -> E + P",
    species_names=['S', 'E', 'C', 'P']
)
```

### Method 2: Programmatic Definition

For full control over rate constant symbols:

```python
from lumping_analysis import Reaction, ReactionNetwork
import sympy as sp

# Define rate constants
k1, km1, k2 = sp.symbols('k_1 k_{-1} k_2', positive=True)

# Define reactions: Reaction(reactant_coeffs, product_coeffs, rate_symbol)
reactions = [
    Reaction((1, 1, 0, 0), (0, 0, 1, 0), k1),   # S + E -> C
    Reaction((0, 0, 1, 0), (1, 1, 0, 0), km1),  # C -> S + E
    Reaction((0, 0, 1, 0), (0, 1, 0, 1), k2),   # C -> E + P
]

network = ReactionNetwork(
    n_species=4,
    reactions=reactions,
    species_names=['S', 'E', 'C', 'P']
)
```

### Method 3: Built-in Networks

Pre-defined networks from the paper:

```python
from lumping_analysis import (
    michaelis_menten_network,      # Example 4.3
    three_species_linear_network,  # Example 4.2
    gpl_replication_network,       # Section 6
    substrate_inhibition_network,
    competitive_inhibition_network,
    double_phosphorylation_network,
)

network = gpl_replication_network()
```

## Core Analysis Functions

### Inspecting the Network

```python
# Get the ODE system dx/dt = F(x,k)
F = network.rhs()

# Get the Jacobian DF(x,k)
J = network.jacobian()

# Get stoichiometric matrix (columns = reaction vectors)
S = network.stoichiometric_matrix()

# Get conservation laws
integrals = network.stoichiometric_first_integrals()

# Print summary
print(network.summary())
```

### Finding Generic Lumpings (Section 3)

Generic lumpings work for **all** parameter values:

```python
analyzer = LumpingAnalyzer(network)
generic = analyzer.find_generic_lumpings()

# Type 1: Based on non-reactant species
print("Common non-reactants:", generic['common_non_reactants'])

# Type 2: Based on stoichiometric first integrals
print("First integrals:", generic['stoichiometric_integrals'])
```

**Theorem (Corollary 3.8):** For generic parameters, exact linear lumping yields only:
- **Type 1:** Elimination of common non-reactant species
- **Type 2:** Projection along stoichiometric first integrals

### Finding Critical Parameters (Section 4)

Critical parameters enable non-trivial lumpings:

```python
import sympy as sp

# Define candidate lumping matrix T
T = sp.Matrix([
    [1, 0, 1],
    [0, 1, 1]
])

# Find critical parameter conditions (Lemma 4.1)
result = analyzer.find_critical_parameters(T)

print("Kernel basis B:")
sp.pprint(result['kernel_basis'])

print("\nConditions for T·DF(x,k)·B = 0:")
for cond in result['conditions']:
    print(f"  {cond} = 0")

print("\nSolutions:")
for sol in result['solutions'].get('solutions', []):
    print(f"  {sol}")
```

### Constrained Lumping (Remark 4.3)

Prescribe some rows of T (e.g., conservation laws):

```python
# Get stoichiometric first integrals
integrals = network.stoichiometric_first_integrals()

# Build T with prescribed rows + free parameters
T, free_params = analyzer.build_constrained_lumping_matrix(
    prescribed_rows=integrals,
    n_free_rows=1
)

# Find critical parameters
result = analyzer.find_critical_parameters(T, free_params=free_params)
```

### Numerical Validation

Verify lumping quality numerically:

```python
import numpy as np

# Define parameter values
k_values = {k1: 1.0, km1: 0.5, k2: 0.8, km2: 0.2}

# Initial conditions
x0 = np.array([1.0, 0.5, 0.0, 0.0])

# Validate
validation = analyzer.validate_numerically(
    T, k_values, x0, 
    t_span=(0, 10),
    tolerance=1e-6
)

print(f"Valid lumping: {validation['is_valid']}")
print(f"Lumping error: {validation['lumping_error']:.2e}")
print(f"Condition error: {validation['condition_error']:.2e}")
```

## LaTeX Export

Generate publication-ready equations:

```python
# Export ODE system
print(network.to_latex())
# Output: \begin{align} \dot{S} &= -k_1 S E + k_{-1} C \\ ... \end{align}

# Export reactions
print(network.reactions_to_latex())
# Output: S + E \xrightarrow{k_1} C \\ ...

# Export critical conditions
print(analyzer.conditions_to_latex(result['conditions']))
```

## Examples from the Paper

### Example 4.2: Three-Species Linear Network

```python
from lumping_analysis import three_species_linear_network, LumpingAnalyzer
import sympy as sp

network = three_species_linear_network()
analyzer = LumpingAnalyzer(network)

# Try lumping to dimension 2
t1, t2 = sp.symbols('t_1 t_2')
T = sp.Matrix([
    [1, 0, t1],
    [0, 1, t2]
])

result = analyzer.find_critical_parameters(T, free_params=[t1, t2])

# Result: t1 + t2 = 1 is required
# Then k_{-2} = k_1 gives a valid lumping
```

### Example 4.3: Michaelis-Menten Constrained Lumping

```python
from lumping_analysis import michaelis_menten_network, LumpingAnalyzer
import sympy as sp

network = michaelis_menten_network()
analyzer = LumpingAnalyzer(network)

# Constrained lumping preserving stoichiometric first integrals
t = sp.Symbol('t')
T = sp.Matrix([
    [1, -1, 0, 1],  # μ₁ = S - E + P
    [0, 1, 1, 0],   # μ₂ = E + C
    [0, 0, 1, t],   # y₃ = C + t·P (parameterized)
])

result = analyzer.find_critical_parameters(T, free_params=[t])

# Result: t = 0 and k₁ = k₋₂ gives critical parameters
# Lumping: y₁ = S + P, y₂ = E + C, y₃ = C
```

### Section 6: Self-Replication Model

```python
from lumping_analysis import gpl_replication_network, LumpingAnalyzer
import sympy as sp

network = gpl_replication_network()
analyzer = LumpingAnalyzer(network)

# Get stoichiometric first integrals
integrals = network.stoichiometric_first_integrals()

# Build constrained lumping matrix
T, params = analyzer.build_constrained_lumping_matrix(
    prescribed_rows=integrals,
    n_free_rows=1
)

result = analyzer.find_critical_parameters(T, free_params=params)

# Critical condition: k₂ = k₄ = 0
# Hidden conservation law: y₁ = P + Iₐ + Iᵦ + r·I
```

## API Reference

### Classes

| Class | Description |
|-------|-------------|
| `Reaction` | Single mass action reaction with stoichiometry and rate constant |
| `ReactionNetwork` | Collection of reactions defining an ODE system |
| `ReactionParser` | Parse reaction strings into networks |
| `LumpingAnalyzer` | Main analysis class for lumping computations |

### Key Methods

| Method | Description |
|--------|-------------|
| `ReactionNetwork.from_string()` | Create network from reaction strings |
| `ReactionNetwork.rhs()` | Get symbolic RHS F(x,k) |
| `ReactionNetwork.jacobian()` | Get Jacobian DF(x,k) |
| `ReactionNetwork.stoichiometric_first_integrals()` | Get conservation laws |
| `LumpingAnalyzer.find_generic_lumpings()` | Find Type 1/2 lumpings |
| `LumpingAnalyzer.find_critical_parameters()` | Find critical k* for given T |
| `LumpingAnalyzer.validate_numerically()` | Numerical validation |
| `LumpingAnalyzer.build_constrained_lumping_matrix()` | Construct T with constraints |

### Factory Functions

| Function | Description |
|----------|-------------|
| `michaelis_menten_network()` | Reversible MM (Example 4.3) |
| `three_species_linear_network()` | Linear chain (Example 4.2) |
| `gpl_replication_network()` | Self-replication (Section 6) |
| `substrate_inhibition_network()` | Substrate inhibition |
| `competitive_inhibition_network()` | Competitive inhibition |
| `double_phosphorylation_network()` | MAPK cascade motif |

## Troubleshooting

### "No solutions found"

The polynomial system may be inconsistent or too complex. Examine conditions directly:

```python
for cond in result['conditions']:
    print(sp.factor(cond))
```

### "Kernel basis is None"

The matrix T has full rank n (no dimension reduction). Reduce the number of rows.

### Memory issues

For large networks, consider analyzing subsystems separately or using numerical methods.

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions are welcome! Please submit issues and pull requests on GitHub.
