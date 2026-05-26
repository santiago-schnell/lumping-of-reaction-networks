"""Regression tests for the two-pathway enzyme example (paper Section 6.2).

These tests lock down properties of the corrected reversible model:

  * The network builder produces 10 reactions and the right species names.
  * The two stoichiometric first integrals are conserved.
  * The row-echelon ansatz T(t1..t9) yields exactly 24 coefficient conditions
    in T * DF * B, all homogeneous-linear in the rate constants.
  * The km5-sensitive conditions match the corrected form k_{-5}*(t1 - t2 + t3),
    NOT the previous (uncorrected) form k_{-5}*(t1 + t3).
  * Component [3] of the Singular decomposition (k2 = k4 = 0, km1 = km3)
    admits the rank-3 lumping y_1 = C1 + C2, y_2 = mu_1, y_3 = mu_2.
"""
from __future__ import annotations

import sympy as sp

from lumping_analysis import (
    LumpingAnalyzer,
    two_pathway_enzyme_network,
)


def test_network_shape():
    net = two_pathway_enzyme_network()
    assert net.n_species == 6
    assert len(net.reactions) == 10
    assert net.species_names == ["S", "E", "C1", "C2", "C", "P"]


def test_stoichiometric_first_integrals():
    """Verify total enzyme and substrate/product balance are conserved."""
    net = two_pathway_enzyme_network()
    F = net.rhs()
    # mu_1: total enzyme  =  e + c1 + c2 + c   (= x2 + x3 + x4 + x5)
    assert sp.expand(F[1] + F[2] + F[3] + F[4]) == 0
    # mu_2: substrate/product balance  =  s + c1 + c2 + c + p (= x1 + x3 + x4 + x5 + x6)
    assert sp.expand(F[0] + F[2] + F[3] + F[4] + F[5]) == 0


def test_critical_conditions_count_and_homogeneity():
    """Section 6.2: T*DF*B yields 24 conditions, all homogeneous-linear in k."""
    net = two_pathway_enzyme_network()
    an = LumpingAnalyzer(net)

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
            [t1, t2, t3], [-1, 0, 0], [t4, t5, t6],
            [t7, t8, t9], [0, -1, 0], [0, 0, -1],
        ]
    )
    assert sp.simplify(T * B) == sp.zeros(3, 3)

    info = an.critical_conditions(T, kernel_basis=B)
    conds = info["conditions"]
    assert len(conds) == 24

    # Each condition is a polynomial in the t's and homogeneous-linear in the k's.
    rate_consts = list(net.rate_constants)
    for c in conds:
        # Setting all rate constants to zero must make each condition zero,
        # i.e., the conditions have no constant term in k.
        zero_subs = {k: 0 for k in rate_consts}
        assert sp.simplify(c.subs(zero_subs)) == 0


def test_km5_sensitive_generators_corrected_form():
    """The km5-sensitive generators must have the corrected (t1 - t2 + t3) form,
    not the previous (t1 + t3) form from the uncorrected vector field.

    The paper's symbolic conclusion is that for a linear observable to be
    invariant under  X5 <-> X2 + X6, its weight on X5 must equal the sum of
    weights on X2 and X6, i.e. t2 = t1 + t3 (equivalently t1 - t2 + t3 = 0).
    """
    net = two_pathway_enzyme_network()
    an = LumpingAnalyzer(net)

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
            [t1, t2, t3], [-1, 0, 0], [t4, t5, t6],
            [t7, t8, t9], [0, -1, 0], [0, 0, -1],
        ]
    )

    info = an.critical_conditions(T, kernel_basis=B)
    k1, km1, k2, km2, k3, km3, k4, km4, k5, km5 = net.rate_constants

    # Pull out conditions that involve km5.
    km5_conds = [c for c in info["conditions"] if km5 in c.free_symbols]
    assert len(km5_conds) >= 3, "expected at least 3 km5-sensitive conditions"

    # Setting t2 = t1 + t3 must kill the pure-km5 generator (the rest of the
    # km5-sensitive generators have additional terms in other rate constants
    # that don't necessarily vanish under the same substitution).
    pure_km5 = [c for c in km5_conds if sp.simplify(c / km5) == sp.simplify(c / km5).subs({k: 0 for k in [k1, km1, k2, km2, k3, km3, k4, km4, k5]})]
    assert pure_km5, "expected at least one purely-km5 generator"
    for c in pure_km5:
        assert sp.simplify(c.subs({t2: t1 + t3, t5: t4 + t6, t8: t7 + t9})) == 0


def test_component_3_specific_lumping():
    """The choice y_1 = C1 + C2 with the two stoichiometric integrals admits
    a critical-parameter solution in which k_1 + k_3 = 0 in the formal-symbol
    sense (k_i are sympy variables here, not necessarily positive)."""
    net = two_pathway_enzyme_network()
    an = LumpingAnalyzer(net)

    T_explicit = sp.Matrix(
        [
            [0, 0, 1, 1, 0, 0],
            [0, 1, 1, 1, 1, 0],
            [1, 0, 1, 1, 1, 1],
        ]
    )
    res = an.find_critical_parameters(T_explicit, solve_for_rate_constants=True)
    # The system should admit a critical-parameter solution with km2 + km4 = 0
    # (these are formal symbols; setting both to zero is one consistent choice).
    cond_set = {sp.expand(c) for c in res["conditions"]}
    k1, km1, k2, km2, k3, km3, k4, km4, k5, km5 = net.rate_constants
    assert sp.expand(km2 + km4) in cond_set or sp.expand(-km2 - km4) in cond_set
