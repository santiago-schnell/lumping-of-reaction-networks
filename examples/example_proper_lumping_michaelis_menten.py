"""Proper lumping for reversible Michaelis--Menten via the column-sum criterion.

This demonstrates the dedicated proper-lumping routine based on the Jacobian
column-sum condition (paper Proposition 5.1).

Run:
    python examples/example_proper_lumping_michaelis_menten.py
"""

from __future__ import annotations

import sympy as sp

from lumping_analysis import LumpingAnalyzer, michaelis_menten_network


def main() -> None:
    net = michaelis_menten_network()
    an = LumpingAnalyzer(net)

    print(net.summary())

    # Partition: {S,P} | {E} | {C}
    res = an.proper_lumping_conditions([["S", "P"], ["E"], ["C"]])

    print("\nProper lumping matrix T:")
    sp.pprint(res["T"])

    print("\nColumn-sum conditions:")
    for c in res["conditions"]:
        print("  ", sp.factor(c), "= 0")


if __name__ == "__main__":
    main()
