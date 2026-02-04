"""Paper Example 4.2: three-species first-order network.

This script reproduces the symbolic setup from the paper and shows how the
critical-parameter equations become a homogeneous linear system in the rate
constants.

Run:
    python examples/example_4_2_three_species.py
"""

from __future__ import annotations

from itertools import combinations

import sympy as sp

from lumping_analysis import LumpingAnalyzer, three_species_linear_network


def main() -> None:
    net = three_species_linear_network()
    an = LumpingAnalyzer(net)

    print(net.summary())

    # Ansatz from the paper: rank-2 lumping matrix with free parameters t1,t2
    t1, t2 = sp.symbols("t1 t2")
    T = sp.Matrix([[1, 0, t1], [0, 1, t2]])

    res = an.find_critical_parameters(T)
    A = res["solutions"].get("linear_system_matrix")
    k_syms = net.rate_constants

    print("\nCritical-parameter conditions (entries of T*A(k)*B):")
    for c in res["conditions"]:
        print("  ", sp.factor(c), "= 0")

    if A is not None:
        print("\nHomogeneous linear system A(t1,t2) * k = 0, with k =", k_syms)
        print("A(t1,t2) =")
        sp.pprint(A)

        # Paper observation: nontrivial solutions exist iff rank(A) < 2.
        # For a 2×4 matrix, rank(A) < 2 is equivalent to all 2×2 minors being zero.
        minors = [sp.factor(A[:, cols].det()) for cols in combinations(range(A.cols), 2)]
        minors = [m for m in minors if m != 0]

        if minors:
            g = sp.factor(sp.gcd_list(minors))
            print("\nGCD of nonzero 2×2 minors (a necessary condition):")
            print("  ", g, "= 0")
            print("\nIn the paper, this reduces to the rank-drop condition t1 + t2 = 1.")
        else:
            print("\nAll 2×2 minors vanish identically (rank(A) < 2 for all t1,t2).")

    # One concrete choice from the paper: t1=1, t2=0 gives the condition k1 = km2.
    print("\nConcrete choice t1=1, t2=0:")
    res_10 = an.find_critical_parameters(T.subs({t1: 1, t2: 0}))
    for c in res_10["conditions"]:
        print("  ", sp.factor(c), "= 0")


if __name__ == "__main__":
    main()
