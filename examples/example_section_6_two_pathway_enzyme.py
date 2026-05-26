"""Paper Section 6.2: two-pathway enzyme mechanism.

This script reproduces the critical-parameter computation of Section 6.2.
It builds the same 3x6 ansatz matrix T(t_1, ..., t_9) that is used in the
manuscript, runs the row-echelon analysis, and demonstrates the analysis
of the irreducible component k_2 = k_4 = 0 (with the additional constraint
k_{-1} = k_{-3} that appears in component [3] of the Singular decomposition).

Run:
    python examples/example_section_6_two_pathway_enzyme.py
"""

from __future__ import annotations

import sympy as sp

from lumping_analysis import (
    LumpingAnalyzer,
    ReductionReportOptions,
    format_reduction_result,
    two_pathway_enzyme_network,
)


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

    # The paper's polynomial kernel basis (lines 1738-1744 of main.tex).
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
    # Sanity check: T * B = 0.
    assert sp.simplify(T * B) == sp.zeros(3, 3), "T . B must vanish"

    info = an.critical_conditions(T, kernel_basis=B)
    conditions = info["conditions"]
    print(f"\nNumber of coefficient conditions in T*DF*B: {len(conditions)}\n")

    print("First 10 factored conditions (compare with Appendix B.2):")
    for c in conditions[:10]:
        print("  ", sp.factor(c), "= 0")

    # ---------------------------------------------------------------
    # Component [3] from the Singular decomposition: k2 = k4 = 0 and km1 = km3.
    # ---------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Component [3] of the critical-parameter decomposition")
    print("(paper Subsection 6.2, lines 1801-1803):")
    print("    k2 = k4 = 0   and   k_{-1} = k_{-3}.")
    print("=" * 60)

    k1, km1, k2, km2, k3, km3, k4, km4, k5, km5 = net.rate_constants
    component_3_subs = {k2: 0, k4: 0, km3: km1}

    remaining = [sp.factor(c.subs(component_3_subs)) for c in conditions]
    nonzero = [r for r in remaining if r != 0]
    print(f"\n{len(conditions) - len(nonzero)} of {len(conditions)} conditions "
          f"vanish identically on this component.")
    print(f"The remaining {len(nonzero)} conditions constrain the t_i's that "
          f"parameterise lumping maps within component [3].")

    # ---------------------------------------------------------------
    # A specific representative of component [3]: y_1 = C1 + C2 (paper line 1849).
    # ---------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Specific lumping in component [3]:  y_1 = x_3 + x_4 = C1 + C2")
    print("=" * 60)
    T_explicit = sp.Matrix(
        [
            [0, 0, 1, 1, 0, 0],   # y_1 = C1 + C2  (the new observable)
            [0, 1, 1, 1, 1, 0],   # y_2 = E + C1 + C2 + C   (total enzyme = mu_1)
            [1, 0, 1, 1, 1, 1],   # y_3 = S + C1 + C2 + C + P  (mu_2)
        ]
    )
    res = an.find_critical_parameters(T_explicit, solve_for_rate_constants=True)
    print(
        format_reduction_result(
            net,
            {**res, "kind": "constrained", "description": "Component [3], y_1 = C1+C2"},
            options=ReductionReportOptions(
                max_conditions=8, max_relations=8, include_T_matrix=False
            ),
        )
    )


if __name__ == "__main__":
    main()
