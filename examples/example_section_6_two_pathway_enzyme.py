"""Paper Section 6.2: two-pathway enzyme mechanism.

This script reproduces the critical-parameter computation of Section 6.2.
It builds the same 3x6 ansatz matrix T(t_1, ..., t_9) used in the manuscript,
runs the row-echelon analysis, and verifies the explicit nontrivial positive-rate
Component [6] reduction.

Component [6] is defined by

    k_2 - k_4 + k_{-1} - k_{-3} = 0,
    k_3 k_{-2} - k_1 k_{-4} = 0.

Run:
    python examples/example_section_6_two_pathway_enzyme.py
"""

from __future__ import annotations

import sympy as sp

from lumping_analysis import LumpingAnalyzer, two_pathway_enzyme_network


def main() -> None:
    net = two_pathway_enzyme_network()
    an = LumpingAnalyzer(net)

    print(net.summary())

    # Paper notation: T(t1,...,t9) is the row-echelon ansatz of Section 6.2.
    t1, t2, t3, t4, t5, t6, t7, t8, t9 = sp.symbols("t1 t2 t3 t4 t5 t6 t7 t8 t9")
    T = sp.Matrix(
        [
            [1, t1, 0, 0, t2, t3],
            [0, t4, 1, 0, t5, t6],
            [0, t7, 0, 1, t8, t9],
        ]
    )

    # Polynomial right-kernel basis corresponding to the ansatz.
    B = sp.Matrix(
        [
            [t1, t2, t3],
            [-1, 0, 0],
            [t4, t5, t6],
            [t7, t8, t9],
            [0, -1, 0],
            [0, 0, -1],
        ]
    )
    assert sp.simplify(T * B) == sp.zeros(3, 3), "T . B must vanish"

    info = an.critical_conditions(T, kernel_basis=B)
    conditions = info["conditions"]
    print(f"\nNumber of coefficient conditions in T*DF*B: {len(conditions)}\n")

    print("First 10 factored conditions (compare with Appendix B.2):")
    for c in conditions[:10]:
        print("  ", sp.factor(c), "= 0")

    # ---------------------------------------------------------------
    # Component [6] from the elimination/minimal-prime output.
    # ---------------------------------------------------------------
    print("\n" + "=" * 68)
    print("Component [6] of the critical-parameter decomposition")
    print("    k2 - k4 + k_{-1} - k_{-3} = 0")
    print("    k3*k_{-2} - k1*k_{-4} = 0")
    print("=" * 68)

    k1, km1, k2, km2, k3, km3, k4, km4, k5, km5 = net.rate_constants
    component_6_subs = {
        km3: km1 + k2 - k4,
        km2: k1 * km4 / k3,
    }

    T_component_6 = sp.Matrix(
        [
            [1, -1, 0, 0, 0, 1],
            [0, k1 / (k1 + k3), 1, 0, k1 / (k1 + k3), 0],
            [0, k3 / (k1 + k3), 0, 1, k3 / (k1 + k3), 0],
        ]
    )

    component_info = an.critical_conditions(T_component_6)
    residuals = [sp.factor(sp.simplify(c.subs(component_6_subs))) for c in component_info["conditions"]]
    print("\nResiduals after substituting Component [6] relations:")
    for r in residuals:
        print("  ", r)
    assert all(r == 0 for r in residuals)

    print("\nLumping matrix T for Component [6]:")
    sp.pprint(T_component_6)

    print("\nLumped variables y = T x:")
    for eq in an.lumped_variable_expressions(T_component_6):
        sp.pprint(eq)

    red = an.construct_reduced_polynomial_system(T_component_6, parameter_subs=component_6_subs)
    print("\nReduced system y' = G(y):")
    sp.pprint(red["G"])

    y1, y2, y3 = red["y_symbols"]
    expected = sp.Matrix(
        [
            0,
            (km1 + k2) * (k1 * y3 - k3 * y2) / (k1 + k3),
            -(km1 + k2) * (k1 * y3 - k3 * y2) / (k1 + k3),
        ]
    )
    assert all(sp.simplify(red["G"][i] - expected[i]) == 0 for i in range(3))
    print("\nVerified expected Component [6] reduced system.")


if __name__ == "__main__":
    main()
