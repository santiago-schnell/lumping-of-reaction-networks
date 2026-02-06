"""Compute a "lumping-adapted" system via a coordinate completion.

Given x' = F(x,k) and a (candidate) lumping matrix T (e×n), we can:

  1) complete T to an invertible n×n matrix T* by appending rows,
  2) define y = T* x, so that the first e coordinates are y_1..y_e = T x,
  3) compute the transformed RHS H(y,k) = T* F(T*^{-1} y, k).

This script demonstrates the workflow on the reversible Michaelis–Menten model.
"""

import sympy as sp

from lumping_analysis import LumpingAnalyzer, michaelis_menten_network


def main() -> None:
    net = michaelis_menten_network()
    an = LumpingAnalyzer(net)

    # A small candidate T from the README quick-start.
    T = sp.Matrix(
        [
            [1, 0, 1, 1],  # S + C + P
            [0, 1, 1, 0],  # E + C
            [0, 0, 1, 0],  # C
        ]
    )

    out = an.lumping_adapted_system(T, y_prefix="y")

    print("Species order:", net.species_names)
    print("\nT:")
    sp.pprint(out["T"])

    print("\nCompleted invertible T*:")
    sp.pprint(out["T_star"])

    print("\nT*^{-1}:")
    sp.pprint(out["T_star_inv"])

    print("\nx expressed in terms of y (x = T*^{-1} y):")
    sp.pprint(out["x_in_terms_of_y"])

    print("\nTransformed RHS H(y,k) = T* F(T*^{-1} y, k):")
    sp.pprint(out["H"])

    print("\nFirst e components (candidate reduced RHS):")
    sp.pprint(out["H_lumped"])

    print("\nDo these depend on the extra coordinates y_{e+1..n}? ->", out["lumped_depends_on_extra"])


if __name__ == "__main__":
    main()
