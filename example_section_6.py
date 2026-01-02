#!/usr/bin/env python3
"""
Section 6: Self-Replication Model Analysis

This script reproduces the analysis from Section 6 of the paper:
"Lumping of reaction networks: Generic settings vs. critical parameters"

Network: GPL biochemical self-replication model from
Gijima and Peacock-López, Mathematics 8, 1042 (2020).

Species: A, B, P (product/template), Iₐ, Iᵦ, I (intermediates)

Key result: At critical parameters k₂ = k₄ = 0, a hidden conservation
law emerges: y₁ = P + Iₐ + Iᵦ + r·I is conserved.
"""

import sympy as sp
from sympy import Matrix, symbols, simplify, factor, Rational
import numpy as np
import sys
sys.path.insert(0, '..')
from lumping_analysis import gpl_replication_network, LumpingAnalyzer

def main():
    print("=" * 60)
    print("Section 6: Self-Replication Model Analysis")
    print("=" * 60)
    
    # Create the network
    network = gpl_replication_network()
    print("\n" + network.summary())
    
    # Create analyzer
    analyzer = LumpingAnalyzer(network)
    
    # ================================================================
    # Part 1: Network Structure
    # ================================================================
    print("\n" + "-" * 40)
    print("Part 1: Network Structure")
    print("-" * 40)
    
    print("\nReaction scheme:")
    print("  A + P ⇌ Iₐ       (k₁, k₋₁)")
    print("  B + Iₐ ⇌ I       (k₂, k₋₂)")
    print("  B + P ⇌ Iᵦ       (k₃, k₋₃)")
    print("  A + Iᵦ ⇌ I       (k₄, k₋₄)")
    print("  I ⇌ 2P           (k₅, k₋₅)")
    
    print("\nAutocatalytic feature: P catalyzes its own formation")
    print("via the intermediate pathways through Iₐ and Iᵦ.")
    
    # ================================================================
    # Part 2: Stoichiometric First Integrals
    # ================================================================
    print("\n" + "-" * 40)
    print("Part 2: Stoichiometric First Integrals")
    print("-" * 40)
    
    integrals = network.stoichiometric_first_integrals()
    print(f"\nNumber of stoichiometric first integrals: {len(integrals)}")
    
    for i, mu in enumerate(integrals):
        terms = []
        species = network.species_names
        for j, coeff in enumerate(mu):
            if coeff != 0:
                if coeff == 1:
                    terms.append(species[j])
                elif coeff == -1:
                    terms.append(f"-{species[j]}")
                else:
                    terms.append(f"{coeff}·{species[j]}")
        integral_str = ' + '.join(terms).replace('+ -', '- ')
        print(f"  μ_{i+1} = {integral_str}")
    
    # ================================================================
    # Part 3: Constrained Lumping Analysis
    # ================================================================
    print("\n" + "-" * 40)
    print("Part 3: Constrained Lumping Analysis")
    print("-" * 40)
    
    print("\nWe seek a lumping matrix T (3×6) where:")
    print("  Row 1: y₁ = P + p·Iₐ + q·Iᵦ + r·I (parameterized)")
    print("  Row 2: μ₁ (stoichiometric first integral)")
    print("  Row 3: μ₂ (stoichiometric first integral)")
    
    # Build the constrained lumping matrix
    # Using p = q = 1 as suggested by symmetry of pathways
    p, q, r = symbols('p q r')
    
    # T matrix with p=q=1, r free
    T = Matrix([
        [0, 0, 1, 1, 1, r],          # y₁ = P + Iₐ + Iᵦ + r·I
        [1, -1, 0, 1, -1, 0],        # μ₁ = A - B + Iₐ - Iᵦ
        [0, 1, 1, 1, 2, 2],          # μ₂ = B + P + Iₐ + 2Iᵦ + 2I
    ])
    
    print("\nLumping matrix T (with p = q = 1, r free):")
    sp.pprint(T)
    
    # Find critical parameters
    result = analyzer.find_critical_parameters(T, free_params=[r])
    
    print("\nConditions from T·DF(x,k)·B = 0:")
    conditions = result['conditions']
    
    # Group and display conditions
    for i, cond in enumerate(conditions[:10]):  # Show first 10
        factored = factor(cond)
        if factored != 0:
            print(f"  ({i+1}) {factored} = 0")
    
    if len(conditions) > 10:
        print(f"  ... ({len(conditions)} conditions total)")
    
    # ================================================================
    # Part 4: Solution Analysis
    # ================================================================
    print("\n" + "-" * 40)
    print("Part 4: Solution Analysis")
    print("-" * 40)
    
    # Define rate constant symbols
    k2 = symbols('k_2')
    k4 = symbols('k_4')
    k5 = symbols('k_5')
    km2 = symbols('k_{-2}')
    km4 = symbols('k_{-4}')
    km5 = symbols('k_{-5}')
    
    print("\nKey observation: Many conditions contain factors of k₂ or k₄.")
    print("\nCase 1: k₂ = k₄ = 0 (critical)")
    print("  - Reactions B + Iₐ → I and A + Iᵦ → I are absent")
    print("  - Full intermediate I can only decompose, not form directly")
    
    # Compute critical r when k2 = k4 = 0 and km5 = 0 (irreversible)
    print("\nWith k₂ = k₄ = k₋₅ = 0, the value of r is determined by:")
    print("  r = (2k₅ + k₋₂ + k₋₄) / (k₅ + k₋₂ + k₋₄)")
    
    # Example numerical values
    k5_val = 1.0
    km2_val = 0.3
    km4_val = 0.2
    
    r_critical = (2*k5_val + km2_val + km4_val) / (k5_val + km2_val + km4_val)
    print(f"\nFor k₅ = {k5_val}, k₋₂ = {km2_val}, k₋₄ = {km4_val}:")
    print(f"  r = {r_critical:.4f}")
    
    # ================================================================
    # Part 5: Hidden Conservation Law
    # ================================================================
    print("\n" + "-" * 40)
    print("Part 5: Hidden Conservation Law")
    print("-" * 40)
    
    print(f"\nAt critical parameters, the quantity:")
    print(f"  y₁ = P + Iₐ + Iᵦ + {r_critical:.4f}·I")
    print(f"\nis CONSERVED (a hidden conservation law).")
    
    print("\nThis is NOT a stoichiometric first integral, but emerges")
    print("only at the critical parameter values.")
    
    print("\nPhysical interpretation:")
    print("  y₁ represents 'total potential product':")
    print("  - Free product P")
    print("  - Product equivalents in intermediates Iₐ, Iᵦ")
    print("  - Weighted contribution from full intermediate I")
    
    # ================================================================
    # Part 6: Numerical Validation
    # ================================================================
    print("\n" + "-" * 40)
    print("Part 6: Numerical Validation")
    print("-" * 40)
    
    # Create T with numerical r
    T_numerical = Matrix([
        [0, 0, 1, 1, 1, Rational(r_critical).limit_denominator(1000)],
        [1, -1, 0, 1, -1, 0],
        [0, 1, 1, 1, 2, 2],
    ]).applyfunc(float)
    
    T_numerical = Matrix([
        [0, 0, 1, 1, 1, r_critical],
        [1, -1, 0, 1, -1, 0],
        [0, 1, 1, 1, 2, 2],
    ])
    
    # Get rate symbols from network
    rate_syms = network.rate_symbols
    # Order: k1, km1, k2, km2, k3, km3, k4, km4, k5, km5
    k1_s, km1_s, k2_s, km2_s, k3_s, km3_s, k4_s, km4_s, k5_s, km5_s = rate_syms
    
    # Critical parameters
    k_critical = {
        k1_s: 1.0, km1_s: 0.5,
        k2_s: 0.0, km2_s: km2_val,  # k₂ = 0 (critical!)
        k3_s: 1.0, km3_s: 0.5,
        k4_s: 0.0, km4_s: km4_val,  # k₄ = 0 (critical!)
        k5_s: k5_val, km5_s: 0.0,
    }
    
    # Initial conditions
    x0 = np.array([2.0, 2.0, 0.5, 0.1, 0.1, 0.5])
    
    validation = analyzer.validate_numerically(
        T_numerical, k_critical, x0, t_span=(0, 10)
    )
    
    print(f"\nAt critical parameters (k₂ = k₄ = 0):")
    print(f"  Valid lumping: {validation['is_valid']}")
    print(f"  Lumping error: {validation['lumping_error']:.2e}")
    print(f"  Condition error: {validation['condition_error']:.2e}")
    
    # Check conservation of y₁
    y_traj = validation['y_trajectory']
    y1_initial = y_traj[0, 0]
    y1_final = y_traj[-1, 0]
    y1_drift = abs(y1_final - y1_initial) / y1_initial * 100
    
    print(f"\nConservation of y₁ = P + Iₐ + Iᵦ + r·I:")
    print(f"  Initial: {y1_initial:.4f}")
    print(f"  Final:   {y1_final:.4f}")
    print(f"  Drift:   {y1_drift:.4f}%")
    
    # Compare with non-critical (epsilon perturbation)
    print("\n--- Comparison with near-critical parameters ---")
    
    epsilons = [0.01, 0.05, 0.1, 0.2]
    
    for eps in epsilons:
        k_near = k_critical.copy()
        k_near[k2_s] = eps
        k_near[k4_s] = eps
        
        val = analyzer.validate_numerically(
            T_numerical, k_near, x0, t_span=(0, 10)
        )
        
        y_traj = val['y_trajectory']
        y1_drift = abs(y_traj[-1, 0] - y_traj[0, 0]) / y_traj[0, 0] * 100
        
        print(f"  ε = k₂ = k₄ = {eps:.2f}: y₁ drift = {y1_drift:.2f}%")
    
    # ================================================================
    # Part 7: Summary
    # ================================================================
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    
    print("""
The GPL self-replication model (6 species, 10 reactions) admits a
constrained lumping at critical parameters k₂ = k₄ = 0.

At these critical values:
  • The hidden conservation law y₁ = P + Iₐ + Iᵦ + r·I emerges
  • The parameter r = (2k₅ + k₋₂ + k₋₄)/(k₅ + k₋₂ + k₋₄) ∈ (1, 2)
  • The 6D system effectively reduces to 3D dynamics

Physical interpretation:
  • Reactions B + Iₐ → I and A + Iᵦ → I are absent
  • The full intermediate I can only decompose (three channels)
  • This asymmetry creates the additional conservation law

For approximate lumping:
  • When k₂, k₄ are small but nonzero, y₁ drifts slowly
  • The drift is O(ε) where ε = max(k₂, k₄)
  • This validates the use of critical parameters for approximate reduction
""")


if __name__ == '__main__':
    main()
