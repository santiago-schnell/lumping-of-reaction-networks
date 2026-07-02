"""Regression tests for the two-pathway enzyme example (paper Section 6.2).

These tests lock down properties of the reversible model and the paper's
Component [6] reduction:

  * the network builder produces 10 reactions and the right species names;
  * the two stoichiometric first integrals are conserved;
  * the row-echelon ansatz T(t1,...,t9) yields exactly 24 coefficient
    conditions in T * DF * B, all homogeneous-linear in the rate constants;
  * the km5-sensitive conditions have the corrected form
    k_{-5}*(t1 - t2 + t3);
  * Component [6], defined by
        k2 - k4 + k_{-1} - k_{-3} = 0,
        k3*k_{-2} - k1*k_{-4} = 0,
    admits the explicit three-dimensional lumping and reduced system reported
    in the manuscript.
"""
from __future__ import annotations

import sympy as sp

from lumping_analysis import (
    LumpingAnalyzer,
    two_pathway_enzyme_network,
)


def _row_echelon_ansatz():
    """Return the Section 6.2 row-echelon ansatz T and polynomial kernel basis B."""
    t1, t2, t3, t4, t5, t6, t7, t8, t9 = sp.symbols("t1 t2 t3 t4 t5 t6 t7 t8 t9")
    T = sp.Matrix(
        [
            [1, t1, 0, 0, t2, t3],
            [0, t4, 1, 0, t5, t6],
            [0, t7, 0, 1, t8, t9],
        ]
    )
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
    return T, B, (t1, t2, t3, t4, t5, t6, t7, t8, t9)


def test_network_shape():
    net = two_pathway_enzyme_network()
    assert net.n_species == 6
    assert len(net.reactions) == 10
    assert net.species_names == ["S", "E", "C1", "C2", "C", "P"]


def test_stoichiometric_first_integrals():
    """Verify total enzyme and substrate/product balance are conserved."""
    net = two_pathway_enzyme_network()
    F = net.rhs()
    # mu_1: total enzyme = e + c1 + c2 + c (= x2 + x3 + x4 + x5).
    assert sp.expand(F[1] + F[2] + F[3] + F[4]) == 0
    # mu_2: substrate/product balance = s + c1 + c2 + c + p.
    assert sp.expand(F[0] + F[2] + F[3] + F[4] + F[5]) == 0


def test_critical_conditions_count_and_homogeneity():
    """Section 6.2: T*DF*B yields 24 conditions, homogeneous-linear in k."""
    net = two_pathway_enzyme_network()
    an = LumpingAnalyzer(net)

    T, B, _ = _row_echelon_ansatz()
    assert sp.simplify(T * B) == sp.zeros(3, 3)

    info = an.critical_conditions(T, kernel_basis=B)
    conds = info["conditions"]
    assert len(conds) == 24

    # Setting all rate constants to zero must make each condition zero.
    zero_subs = {k: 0 for k in net.rate_constants}
    for c in conds:
        assert sp.simplify(c.subs(zero_subs)) == 0


def test_km5_sensitive_generators_corrected_form():
    """The km5-sensitive generators must have the corrected (t1 - t2 + t3) form."""
    net = two_pathway_enzyme_network()
    an = LumpingAnalyzer(net)

    T, B, ts = _row_echelon_ansatz()
    t1, t2, t3, t4, t5, t6, t7, t8, t9 = ts
    info = an.critical_conditions(T, kernel_basis=B)
    k1, km1, k2, km2, k3, km3, k4, km4, k5, km5 = net.rate_constants

    km5_conds = [c for c in info["conditions"] if km5 in c.free_symbols]
    assert len(km5_conds) >= 3, "expected at least 3 km5-sensitive conditions"

    other_rates = [k1, km1, k2, km2, k3, km3, k4, km4, k5]
    pure_km5 = [
        c
        for c in km5_conds
        if sp.simplify(c / km5)
        == sp.simplify(c / km5).subs({k: 0 for k in other_rates})
    ]
    assert pure_km5, "expected at least one purely-km5 generator"
    for c in pure_km5:
        assert sp.simplify(c.subs({t2: t1 + t3, t5: t4 + t6, t8: t7 + t9})) == 0


def test_component_6_specific_lumping_and_reduced_system():
    """Verify the paper's Component [6] explicit lumping and reduced ODE."""
    net = two_pathway_enzyme_network()
    an = LumpingAnalyzer(net)
    k1, km1, k2, km2, k3, km3, k4, km4, k5, km5 = net.rate_constants

    T_component_6 = sp.Matrix(
        [
            [1, -1, 0, 0, 0, 1],
            [0, k1 / (k1 + k3), 1, 0, k1 / (k1 + k3), 0],
            [0, k3 / (k1 + k3), 0, 1, k3 / (k1 + k3), 0],
        ]
    )

    # Component [6] conditions:
    #   k2 - k4 + km1 - km3 = 0,
    #   k3*km2 - k1*km4 = 0.
    component_6_subs = {
        km3: km1 + k2 - k4,
        km2: k1 * km4 / k3,
    }

    info = an.critical_conditions(T_component_6)
    assert info["conditions"], "the generic symbolic T should impose component conditions"
    assert all(sp.simplify(c.subs(component_6_subs)) == 0 for c in info["conditions"])

    red = an.construct_reduced_polynomial_system(
        T_component_6,
        parameter_subs=component_6_subs,
    )
    y1, y2, y3 = red["y_symbols"]
    G = red["G"]
    expected = sp.Matrix(
        [
            0,
            (km1 + k2) * (k1 * y3 - k3 * y2) / (k1 + k3),
            -(km1 + k2) * (k1 * y3 - k3 * y2) / (k1 + k3),
        ]
    )
    assert G.shape == (3, 1)
    assert all(sp.simplify(G[i] - expected[i]) == 0 for i in range(3))
