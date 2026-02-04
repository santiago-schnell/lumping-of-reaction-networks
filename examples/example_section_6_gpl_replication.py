"""Paper Section 6: self-replication model (Gijima--Peacock-Lopez).

This script shows how to set up the constrained lumping ansatz from the paper
and compute the critical-parameter linear system in the rate constants.

Run:
    python examples/example_section_6_gpl_replication.py
"""

from __future__ import annotations

import sympy as sp

from lumping_analysis import (
    LumpingAnalyzer,
    ReductionReportOptions,
    format_reduction_result,
    gpl_replication_network,
)


def main() -> None:
    net = gpl_replication_network()
    an = LumpingAnalyzer(net)

    print(net.summary())

    # Paper notation: y1 = x3 + p x4 + q x5 + r x6
    p, q, r = sp.symbols("p q r")
    T = sp.Matrix(
        [
            [0, 0, 1, p, q, r],
            [1, -1, 0, 1, -1, 0],
            [0, 1, 1, 1, 2, 2],
        ]
    )

    res = an.find_critical_parameters(T, solve_for_rate_constants=True)
    res.update({"kind": "constrained", "description": "Section 6 ansatz (p,q,r)"})

    print(
        format_reduction_result(
            net,
            res,
            options=ReductionReportOptions(max_conditions=12, max_relations=12, include_T_matrix=False),
        )
    )

    print("\nNow substitute the symmetric choice p=q=1 (as discussed in the paper):")
    res_11 = an.find_critical_parameters(T.subs({p: 1, q: 1}), solve_for_rate_constants=True)
    res_11.update({"kind": "constrained", "description": "Section 6 ansatz with p=q=1"})
    print(
        format_reduction_result(
            net,
            res_11,
            options=ReductionReportOptions(max_conditions=12, max_relations=12, include_T_matrix=False),
        )
    )


if __name__ == "__main__":
    main()
