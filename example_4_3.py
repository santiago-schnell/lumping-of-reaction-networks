#!/usr/bin/env python3
"""
Example 4.3: Michaelis-Menten Constrained Lumping

This script reproduces the analysis from Example 4.3 of the paper:
"Lumping of reaction networks: Generic settings vs. critical parameters"

Network:  S + E ⇌ C ⇌ E + P

We analyze constrained lumping where the stoichiometric first integrals
are preserved, and find critical parameters for additional reduction.
"""

import sympy as sp
from sympy import Matrix, symbols, simplify, factor, latex
import sys
sys.path.insert(0, '..')
from lumping_analysis import michaelis_menten_network, LumpingAnalyzer

def main():
    print("=" * 60)
    print("Example 4.3: Michaelis-Menten Constrained Lumping")
    print("=" * 60)
    
    # Create the network
    network = michaelis_menten_network()
    print("\n" + network.summary())
    
    # Show the ODE system
    print("\nODE system:")
    F = network.rhs()
    for i, name in enumerate(network.species_names):
        print(f"  d{name}/dt = {F[i]}")
    
    # Create analyzer
    analyzer = LumpingAnalyzer(network)
    
    # ================================================================
    # Part 1: Stoichiometric First Integrals
    # ================================================================
    print("\n" + "-" * 40)
    print("Part 1: Stoichiometric First Integrals")
    print("-" * 40)
    
    integrals = network.stoichiometric_first_integrals()
    print(f"\nThe network has {len(integrals)} stoichiometric first integrals:")
    
    # Standard form
    print("\n  μ₁ = S - E + P  (substrate-product balance)")
    print("  μ₂ = E + C      (total enzyme conservation)")
    
    # ================================================================
    # Part 2: Constrained Lumping Setup
    # ================================================================
    print("\n" + "-" * 40)
    print("Part 2: Constrained Lumping Setup (Remark 4.3)")
    print("-" * 40)
    
    print("\nWe seek a 3×4 lumping matrix T where:")
    print("  Row 1: μ₁ = S - E + P")
    print("  Row 2: μ₂ = E + C")
    print("  Row 3: parameterized as C + t·P")
    
    t = sp.Symbol('t')
    
    # T matrix as in Example 4.3
    T = Matrix([
        [1, -1, 0, 1],  # μ₁ = S - E + P
        [0, 1, 1, 0],   # μ₂ = E + C
        [0, 0, 1, t],   # y₃ = C + t·P
    ])
    
    print("\nLumping matrix T:")
    sp.pprint(T)
    
    # ================================================================
    # Part 3: Find Critical Parameters
    # ================================================================
    print("\n" + "-" * 40)
    print("Part 3: Critical Parameter Analysis")
    print("-" * 40)
    
    result = analyzer.find_critical_parameters(T, free_params=[t])
    
    print("\nKernel basis B:")
    sp.pprint(result['kernel_basis'])
    
    print("\nConditions from T·DF(x,k)·B = 0:")
    for i, cond in enumerate(result['conditions']):
        factored = factor(cond)
        print(f"  ({i+1}) {factored} = 0")
    
    # ================================================================
    # Part 4: Analysis of Conditions
    # ================================================================
    print("\n" + "-" * 40)
    print("Part 4: Analysis of Conditions")
    print("-" * 40)
    
    print("\nThe conditions factor as:")
    print("  - Terms with (1 + t₁ - t₂) or similar factors")
    print("  - Setting t = 0 simplifies significantly")
    
    # Substitute t = 0
    T_t0 = T.subs(t, 0)
    print("\nWith t = 0, the lumping matrix becomes:")
    sp.pprint(T_t0)
    
    result_t0 = analyzer.find_critical_parameters(T_t0)
    
    print("\nRemaining conditions with t = 0:")
    for cond in result_t0['conditions']:
        factored = factor(cond)
        if factored != 0:
            print(f"  {factored} = 0")
    
    print("\nThe key condition is: k₁ = k₋₂")
    print("(Forward binding rate equals reverse product release rate)")
    
    # ================================================================
    # Part 5: The Reduced System
    # ================================================================
    print("\n" + "-" * 40)
    print("Part 5: The Reduced System at Critical Parameters")
    print("-" * 40)
    
    print("\nAt critical parameters (t = 0, k₁ = k₋₂):")
    print("  y₁ = S - E + P  (constant)")
    print("  y₂ = E + C      (constant)")
    print("  y₃ = C")
    
    print("\nThe reduced system has effectively ONE dynamic variable (C),")
    print("with S, E, P determined by the conservation laws.")
    
    # Show the reduced ODE
    print("\nReduced dynamics for y₃ = C:")
    print("  dy₃/dt = k₁·S·E - (k₋₁ + k₂)·C + k₋₂·E·P")
    print("\nSubstituting S = y₁ + E - P and using k₁ = k₋₂:")
    print("  dy₃/dt = k₁·(y₁ + E - P)·E - (k₋₁ + k₂)·y₃ + k₁·E·P")
    print("         = k₁·E·(y₁ + E) - (k₋₁ + k₂)·y₃")
    
    # ================================================================
    # Part 6: Numerical Validation
    # ================================================================
    print("\n" + "-" * 40)
    print("Part 6: Numerical Validation")
    print("-" * 40)
    
    import numpy as np
    
    # Get rate constant symbols from network
    k1, km1, k2, km2 = network.rate_symbols
    
    # Critical parameters
    k_critical = {k1: 1.0, km1: 0.5, k2: 0.3, km2: 1.0}
    
    # Initial conditions: S=2, E=1, C=0, P=0
    x0 = np.array([2.0, 1.0, 0.0, 0.0])
    
    validation = analyzer.validate_numerically(
        T_t0, k_critical, x0, t_span=(0, 20)
    )
    
    print(f"\nAt critical parameters (k₁ = k₋₂ = 1.0):")
    print(f"  Valid lumping: {validation['is_valid']}")
    print(f"  Lumping error: {validation['lumping_error']:.2e}")
    print(f"  Condition error: {validation['condition_error']:.2e}")
    
    # Verify conservation laws
    y_traj = validation['y_trajectory']
    print(f"\nConservation law verification:")
    print(f"  y₁ (S-E+P): initial = {y_traj[0,0]:.4f}, final = {y_traj[-1,0]:.4f}")
    print(f"  y₂ (E+C):   initial = {y_traj[0,1]:.4f}, final = {y_traj[-1,1]:.4f}")
    print(f"  y₃ (C):     initial = {y_traj[0,2]:.4f}, final = {y_traj[-1,2]:.4f}")
    
    # Non-critical for comparison
    k_noncritical = {k1: 1.0, km1: 0.5, k2: 0.3, km2: 0.5}
    
    validation_nc = analyzer.validate_numerically(
        T_t0, k_noncritical, x0, t_span=(0, 20)
    )
    
    print(f"\nAt non-critical parameters (k₁ = 1.0, k₋₂ = 0.5):")
    print(f"  Valid lumping: {validation_nc['is_valid']}")
    print(f"  Lumping error: {validation_nc['lumping_error']:.2e}")
    
    # ================================================================
    # Part 7: LaTeX Export
    # ================================================================
    print("\n" + "-" * 40)
    print("Part 7: LaTeX Export")
    print("-" * 40)
    
    print("\nReactions in LaTeX:")
    print(network.reactions_to_latex())
    
    print("\n" + "=" * 60)
    print("Conclusion: The Michaelis-Menten system admits constrained")
    print("lumping preserving enzyme conservation when k₁ = k₋₂.")
    print("This reduces the 4D system to effectively 1D dynamics.")
    print("=" * 60)


if __name__ == '__main__':
    main()
