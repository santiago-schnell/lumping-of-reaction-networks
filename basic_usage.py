"""Basic usage demo for the lumping_analysis package.

This is a short, repo-level script intended for quick experimentation. For more
focused scripts, see the files in the `examples/` directory.

Run:
    python basic_usage.py
"""

from __future__ import annotations

import sympy as sp

from lumping_analysis import (
    LumpingAnalyzer,
    ReactionNetwork,
    michaelis_menten_network,
    three_species_linear_network,
)


def main() -> None:
    # ------------------------------------------------------------------
    # 1) Define a network from a string
    # ------------------------------------------------------------------
    net = ReactionNetwork.from_string("A + B <->[k1][km1] C", species_names=["A", "B", "C"])
    print(net.summary())

    # ------------------------------------------------------------------
    # 2) Generic lumpings
    # ------------------------------------------------------------------
    an = LumpingAnalyzer(net)
    gen = an.find_generic_lumpings()
    print("\nGeneric lumpings:")
    print("  Common non-reactants:", gen["common_non_reactants"])
    print("  Stoichiometric first integrals:")
    for mu in gen["stoichiometric_integrals"]:
        print("   ", mu)

    # ------------------------------------------------------------------
    # 3) Critical parameters (paper Example 4.2)
    # ------------------------------------------------------------------
    net3 = three_species_linear_network()
    an3 = LumpingAnalyzer(net3)
    t1, t2 = sp.symbols("t1 t2")
    T = sp.Matrix([[1, 0, t1], [0, 1, t2]])
    res = an3.find_critical_parameters(T.subs({t1: 1, t2: 0}))
    print("\nThree-species example, t1=1, t2=0:")
    for c in res["conditions"]:
        print("  ", sp.factor(c), "= 0")

    # ------------------------------------------------------------------
    # 4) Proper lumping (paper Section 5)
    # ------------------------------------------------------------------
    mm = michaelis_menten_network()
    amm = LumpingAnalyzer(mm)
    pl = amm.proper_lumping_conditions([["S", "P"], ["E"], ["C"]])
    print("\nProper lumping for Michaelis--Menten (blocks {S,P}|{E}|{C}):")
    print("T =")
    sp.pprint(pl["T"])
    print("Conditions:")
    for c in pl["conditions"]:
        print("  ", sp.factor(c), "= 0")


if __name__ == "__main__":
    main()
