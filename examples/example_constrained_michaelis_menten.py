"""Paper constrained example: reversible Michaelis--Menten preserving integrals.

This script implements the constrained ansatz from Example \label{ex:mm} in the
Feb 4, 2026 draft.

Run:
    python examples/example_constrained_michaelis_menten.py
"""

from __future__ import annotations

import sympy as sp

from lumping_analysis import LumpingAnalyzer, michaelis_menten_network


def main() -> None:
    net = michaelis_menten_network()
    an = LumpingAnalyzer(net)

    print(net.summary())

    k1, km1, k2, km2 = net.rate_constants

    # Constrained ansatz (paper Example ex:mm)
    t = sp.Symbol("t")
    T = sp.Matrix(
        [
            [1, 0, 1, 1],  # s + c + p
            [0, 1, 1, 0],  # e + c
            [0, 0, 1, t],  # c + t p
        ]
    )

    res = an.find_critical_parameters(T)
    print("\nSymbolic conditions (general t):")
    for c in res["conditions"]:
        print("  ", sp.factor(c), "= 0")

    # Inspect special cases highlighted in the paper.
    for tval in [0, 1]:
        print(f"\n--- Substituting t = {tval} ---")
        res_tv = an.find_critical_parameters(T.subs({t: tval}))
        for c in res_tv["conditions"]:
            print("  ", sp.factor(c), "= 0")

    # For t=0, the paper highlights the distinguished condition k1 = k_{-2}.
    # Our notation: k_{-2} <-> km2.
    print("\n--- Constructing a reduced polynomial system for t=0 and km2=k1 ---")
    T0 = sp.Matrix(
        [
            [1, 0, 0, 1],  # y1 = s + p
            [0, 1, 1, 0],  # y2 = e + c
            [1, 0, 1, 1],  # y3 = s + c + p
        ]
    )
    red = an.construct_reduced_polynomial_system(T0, parameter_subs={km2: k1})
    print("y' = G(y) with y = T x:")
    sp.pprint(red["G"])


if __name__ == "__main__":
    main()
