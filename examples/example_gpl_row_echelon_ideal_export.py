"""Row-echelon ansatz + ideal export (GPL/self-replication example).

This script mirrors the standard CAS workflow:

  1) choose a row-echelon ansatz for a lumping matrix T,
  2) build a *polynomial* kernel basis B for ker(T),
  3) compute the Jacobian invariance matrix T*DF(x,k)*B,
  4) extract coefficients in the state variables x,
  5) treat those coefficient polynomials as generators of an ideal,
  6) export the ideal to Singular for Groebner / primary decomposition.

Run from the repository root:

    python examples/example_gpl_row_echelon_ideal_export.py
"""

import sympy as sp

from lumping_analysis import LumpingAnalyzer, gpl_replication_network


def main() -> None:
    net = gpl_replication_network()
    an = LumpingAnalyzer(net)

    # Match the attached notebook: e=3 pivot columns [0,1,2] and free parameters t1..t9.
    ansatz = an.row_echelon_ansatz(e=3, pivot_cols=[0, 1, 2], free_param_prefix="t", flat_numbering=True)
    T = ansatz["T"]
    B = ansatz["B"]

    print("Species order:", net.species_names)
    print("\nRow-echelon ansatz T (3×6):")
    sp.pprint(T)

    print("\nPolynomial kernel basis B for ker(T) (6×3):")
    sp.pprint(B)

    res = an.find_critical_parameters(T, kernel_basis=B, solve_for_rate_constants=True)

    print("\nNumber of coefficient conditions:", len(res["conditions"]))
    print("Conditions (factored):")
    for c in res["conditions"]:
        print("  ", sp.factor(c), "= 0")

    # Package as an ideal for Singular.
    I = an.critical_ideal(T, kernel_basis=B)

    print("\n--- Singular script (ring + ideal) ---")
    print(I.to_singular_script(comment="GPL example: critical-parameter ideal from T*DF(x,k)*B = 0"))

    # If you have Singular installed and on PATH, you can also run:
    # out = I.run(timeout=120)
    # print(out)


if __name__ == "__main__":
    main()
