#!/usr/bin/env python3
"""
Basic Usage Examples for lumping_analysis module

This script demonstrates the core features of the lumping analysis module
with simple, self-contained examples.
"""

import sympy as sp
from sympy import Matrix, symbols
import numpy as np
import sys
sys.path.insert(0, '..')
from lumping_analysis import (
    ReactionNetwork, Reaction, LumpingAnalyzer,
    michaelis_menten_network, three_species_linear_network,
    list_available_networks
)


def example_1_string_parser():
    """Demonstrate the reaction string parser."""
    print("\n" + "=" * 50)
    print("Example 1: Creating Networks from Strings")
    print("=" * 50)
    
    # Method 1: Simple string
    network = ReactionNetwork.from_string("A + B <-> C")
    print("\nFrom 'A + B <-> C':")
    print(f"  Species: {network.species_names}")
    print(f"  Reactions: {len(network.reactions)}")
    
    # Method 2: Multiple reactions
    network = ReactionNetwork.from_string("""
        S + E <-> C
        C -> E + P
    """)
    print("\nFrom 'S + E <-> C; C -> E + P':")
    print(f"  Species: {network.species_names}")
    print(f"  Reactions: {len(network.reactions)}")
    for i, rxn in enumerate(network.reactions):
        print(f"    {i+1}. {rxn.to_string(network.species_names)}")
    
    # Method 3: With stoichiometric coefficients
    network = ReactionNetwork.from_string("2A -> B; B + C <-> D")
    print("\nFrom '2A -> B; B + C <-> D':")
    for i, rxn in enumerate(network.reactions):
        print(f"    {i+1}. {rxn.to_string(network.species_names)}")


def example_2_built_in_networks():
    """Demonstrate built-in network factory functions."""
    print("\n" + "=" * 50)
    print("Example 2: Built-in Networks")
    print("=" * 50)
    
    print("\nAvailable networks:")
    for name, desc in list_available_networks().items():
        print(f"  {name}(): {desc}")
    
    # Load Michaelis-Menten
    network = michaelis_menten_network()
    print("\n" + network.summary())


def example_3_generic_lumping():
    """Demonstrate finding generic lumpings."""
    print("\n" + "=" * 50)
    print("Example 3: Generic Lumping Analysis")
    print("=" * 50)
    
    network = michaelis_menten_network()
    analyzer = LumpingAnalyzer(network)
    
    generic = analyzer.find_generic_lumpings()
    
    print("\nGeneric lumping analysis for Michaelis-Menten:")
    print(f"  Common non-reactants: {generic['common_non_reactants']}")
    print(f"  Type 1 possible: {generic['type1_possible']}")
    print(f"  Type 2 possible: {generic['type2_possible']}")
    
    print(f"\n  Stoichiometric first integrals ({len(generic['stoichiometric_integrals'])}):")
    for i, mu in enumerate(generic['stoichiometric_integrals']):
        print(f"    μ_{i+1} = {mu.T}")


def example_4_critical_parameters():
    """Demonstrate finding critical parameters."""
    print("\n" + "=" * 50)
    print("Example 4: Finding Critical Parameters")
    print("=" * 50)
    
    network = three_species_linear_network()
    analyzer = LumpingAnalyzer(network)
    
    print("\nNetwork: X₁ ⇌ X₂ ⇌ X₃")
    
    # Define a candidate lumping matrix
    T = Matrix([
        [1, 0, 1],  # y₁ = x₁ + x₃
        [0, 1, 0]   # y₂ = x₂
    ])
    
    print("\nCandidate lumping matrix T:")
    print("  y₁ = x₁ + x₃")
    print("  y₂ = x₂")
    
    result = analyzer.find_critical_parameters(T)
    
    print("\nCritical parameter conditions:")
    for cond in result['conditions']:
        print(f"  {sp.factor(cond)} = 0")
    
    print("\nInterpretation: k₁ = k₋₂ gives a valid lumping")


def example_5_numerical_validation():
    """Demonstrate numerical validation."""
    print("\n" + "=" * 50)
    print("Example 5: Numerical Validation")
    print("=" * 50)
    
    network = three_species_linear_network()
    analyzer = LumpingAnalyzer(network)
    
    T = Matrix([
        [1, 0, 1],
        [0, 1, 0]
    ])
    
    # Get rate constant symbols FROM THE NETWORK
    k1, km1, k2, km2 = network.rate_symbols
    
    # Critical parameters (k₁ = k₋₂)
    k_critical = {k1: 1.0, km1: 0.5, k2: 0.8, km2: 1.0}
    x0 = np.array([1.0, 0.5, 0.3])
    
    print("\nValidating at CRITICAL parameters (k₁ = k₋₂ = 1.0):")
    result = analyzer.validate_numerically(T, k_critical, x0)
    print(f"  Valid: {result['is_valid']}")
    print(f"  Condition error (T·DF·B): {result['condition_error']:.2e}")
    print(f"  Lumping error: {result['lumping_error']:.2e}")
    
    # Non-critical parameters
    k_noncritical = {k1: 1.0, km1: 0.5, k2: 0.8, km2: 0.3}
    
    print("\nValidating at NON-CRITICAL parameters (k₁ = 1.0, k₋₂ = 0.3):")
    result = analyzer.validate_numerically(T, k_noncritical, x0)
    print(f"  Valid: {result['is_valid']}")
    print(f"  Condition error (T·DF·B): {result['condition_error']:.2e}")
    print(f"  Lumping error: {result['lumping_error']:.2e}")
    print("\n  Note: The condition error reflects |k₁ - k₋₂|")


def example_6_latex_export():
    """Demonstrate LaTeX export."""
    print("\n" + "=" * 50)
    print("Example 6: LaTeX Export")
    print("=" * 50)
    
    network = michaelis_menten_network()
    
    print("\nReactions in LaTeX:")
    print("-" * 40)
    print(network.reactions_to_latex())
    
    print("\nODE system in LaTeX:")
    print("-" * 40)
    print(network.to_latex())


def example_7_constrained_lumping():
    """Demonstrate constrained lumping matrix construction."""
    print("\n" + "=" * 50)
    print("Example 7: Constrained Lumping")
    print("=" * 50)
    
    network = michaelis_menten_network()
    analyzer = LumpingAnalyzer(network)
    
    # Get stoichiometric first integrals
    integrals = network.stoichiometric_first_integrals()
    
    print(f"\nPrescribed rows (stoichiometric first integrals): {len(integrals)}")
    
    # Build constrained lumping matrix
    T, free_params = analyzer.build_constrained_lumping_matrix(
        prescribed_rows=integrals,
        n_free_rows=1
    )
    
    print("\nConstrained lumping matrix T:")
    sp.pprint(T)
    
    print(f"\nFree parameters: {free_params}")
    
    # Find critical parameters
    result = analyzer.find_critical_parameters(T, free_params=free_params)
    
    print(f"\nNumber of critical conditions: {len(result['conditions'])}")


def main():
    """Run all examples."""
    print("=" * 50)
    print("LUMPING ANALYSIS MODULE - BASIC USAGE EXAMPLES")
    print("=" * 50)
    
    example_1_string_parser()
    example_2_built_in_networks()
    example_3_generic_lumping()
    example_4_critical_parameters()
    example_5_numerical_validation()
    example_6_latex_export()
    example_7_constrained_lumping()
    
    print("\n" + "=" * 50)
    print("All examples completed successfully!")
    print("=" * 50)


if __name__ == '__main__':
    main()
