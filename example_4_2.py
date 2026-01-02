#!/usr/bin/env python3
"""
Example 4.2: Three-Species Linear Network

This script reproduces the analysis from Example 4.2 of the paper:
"Lumping of reaction networks: Generic settings vs. critical parameters"

Network:  X₁ ⇌ X₂ ⇌ X₃

The network admits the stoichiometric first integral μ = X₁ + X₂ + X₃.
We analyze under what conditions a rank-2 lumping exists.
"""

import sympy as sp
from sympy import Matrix, symbols, simplify, factor, solve
import sys
sys.path.insert(0, '..')
from lumping_analysis import three_species_linear_network, LumpingAnalyzer

def main():
    print("=" * 60)
    print("Example 4.2: Three-Species Linear Network")
    print("=" * 60)
    
    # Create the network
    network = three_species_linear_network()
    print("\n" + network.summary())
    
    # Create analyzer
    analyzer = LumpingAnalyzer(network)
    
    # ================================================================
    # Part 1: Generic Lumping Analysis
    # ================================================================
    print("\n" + "-" * 40)
    print("Part 1: Generic Lumping Analysis")
    print("-" * 40)
    
    generic = analyzer.find_generic_lumpings()
    
    print(f"\nCommon non-reactant species: {generic['common_non_reactants']}")
    print("(Empty set means no Type 1 lumping exists)")
    
    print(f"\nStoichiometric first integrals: {len(generic['stoichiometric_integrals'])}")
    for i, mu in enumerate(generic['stoichiometric_integrals']):
        print(f"  μ_{i+1} = {mu.T}")
    
    # ================================================================
    # Part 2: Critical Parameter Analysis
    # ================================================================
    print("\n" + "-" * 40)
    print("Part 2: Critical Parameter Analysis")
    print("-" * 40)
    
    # Ansatz: T = [1, 0, t₁; 0, 1, t₂] for lumping to dimension 2
    t1, t2 = symbols('t_1 t_2')
    T = Matrix([
        [1, 0, t1],
        [0, 1, t2]
    ])
    
    print("\nCandidate lumping matrix T (with free parameters t₁, t₂):")
    sp.pprint(T)
    print("\nThis gives y₁ = x₁ + t₁·x₃ and y₂ = x₂ + t₂·x₃")
    
    # Find critical parameters
    result = analyzer.find_critical_parameters(T, free_params=[t1, t2])
    
    print("\nKernel basis B (ker T):")
    sp.pprint(result['kernel_basis'])
    
    print("\nConditions from T·DF(x,k)·B = 0:")
    for i, cond in enumerate(result['conditions']):
        factored = factor(cond)
        print(f"  ({i+1}) {factored} = 0")
    
    # ================================================================
    # Part 3: Solving the Conditions
    # ================================================================
    print("\n" + "-" * 40)
    print("Part 3: Solving the Conditions")
    print("-" * 40)
    
    # The key insight: conditions factor as products
    # We need t₁ + t₂ = 1 from the structure
    print("\nFrom the conditions, we find that t₁ + t₂ = 1 is required.")
    print("Setting t₂ = 1 - t₁, the remaining condition involves rate constants.")
    
    # Substitute t2 = 1 - t1
    conditions_sub = [c.subs(t2, 1 - t1) for c in result['conditions']]
    conditions_sub = [simplify(c) for c in conditions_sub]
    conditions_sub = [c for c in conditions_sub if c != 0]
    
    print("\nConditions after substituting t₂ = 1 - t₁:")
    for cond in conditions_sub[:5]:  # Show first few
        print(f"  {factor(cond)} = 0")
    
    # ================================================================
    # Part 4: Specific Example
    # ================================================================
    print("\n" + "-" * 40)
    print("Part 4: Specific Example (t₁ = 1, t₂ = 0)")
    print("-" * 40)
    
    T_specific = Matrix([
        [1, 0, 1],
        [0, 1, 0]
    ])
    
    print("\nWith t₁ = 1, t₂ = 0:")
    print("  y₁ = x₁ + x₃")
    print("  y₂ = x₂")
    
    result_specific = analyzer.find_critical_parameters(T_specific)
    
    print("\nCritical parameter conditions:")
    for cond in result_specific['conditions']:
        print(f"  {factor(cond)} = 0")
    
    # Get rate constant symbols
    k1, km1, k2, km2 = symbols('k_1 k_{-1} k_2 k_{-2}')
    
    print("\nThe condition k₋₂ = k₁ (or equivalently, the first and last")
    print("rate constants are equal) yields a valid lumping.")
    
    # ================================================================
    # Part 5: Verify with Numerical Validation
    # ================================================================
    print("\n" + "-" * 40)
    print("Part 5: Numerical Validation")
    print("-" * 40)
    
    import numpy as np
    
    # Get rate symbols from network
    k1, km1, k2, km2 = network.rate_symbols
    
    # Parameters satisfying critical condition
    k_critical = {
        k1: 1.0,
        km1: 0.5,
        k2: 0.8,
        km2: 1.0,  # = k_1 (critical!)
    }
    
    x0 = np.array([1.0, 0.5, 0.3])
    
    validation = analyzer.validate_numerically(
        T_specific, k_critical, x0, t_span=(0, 20)
    )
    
    print(f"\nAt critical parameters (k₁ = k₋₂ = 1.0):")
    print(f"  Valid lumping: {validation['is_valid']}")
    print(f"  Lumping error: {validation['lumping_error']:.2e}")
    print(f"  Condition error: {validation['condition_error']:.2e}")
    
    # Compare with non-critical parameters
    k_noncritical = {
        k1: 1.0,
        km1: 0.5,
        k2: 0.8,
        km2: 0.3,  # ≠ k_1 (not critical)
    }
    
    validation_nc = analyzer.validate_numerically(
        T_specific, k_noncritical, x0, t_span=(0, 20)
    )
    
    print(f"\nAt non-critical parameters (k₁ = 1.0, k₋₂ = 0.3):")
    print(f"  Valid lumping: {validation_nc['is_valid']}")
    print(f"  Lumping error: {validation_nc['lumping_error']:.2e}")
    print(f"  Condition error: {validation_nc['condition_error']:.2e}")
    
    print("\n" + "=" * 60)
    print("Conclusion: The lumping y = (x₁ + x₃, x₂)ᵀ is exact when")
    print("k₁ = k₋₂, and approximate (with O(|k₁ - k₋₂|) error) otherwise.")
    print("=" * 60)


if __name__ == '__main__':
    main()
