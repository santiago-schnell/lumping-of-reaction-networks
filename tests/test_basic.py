import sympy as sp

from lumping_analysis import (
    ReactionNetwork,
    LumpingAnalyzer,
    michaelis_menten_network,
    three_species_linear_network,
)


def test_stoichiometric_integrals_are_valid():
    net = michaelis_menten_network()
    S = net.stoichiometric_matrix()
    for mu in net.stoichiometric_first_integrals(integer_basis=True):
        assert mu.shape == (1, net.n_species)
        assert (mu * S) == sp.zeros(1, S.cols)


def test_three_species_critical_condition_matches_paper_example():
    net = three_species_linear_network()
    an = LumpingAnalyzer(net)

    # Paper Example 4.2 special case t1=1,t2=0 gives k_{-2} = k_1.
    T = sp.Matrix([[1, 0, 1], [0, 1, 0]])
    res = an.find_critical_parameters(T)

    k1, km1, k2, km2 = net.rate_constants
    expected = sp.simplify(k1 - km2)

    # We expect the condition k1-km2=0 (up to sign).
    assert any(sp.simplify(c - expected) == 0 or sp.simplify(c + expected) == 0 for c in res["conditions"])


def test_generic_type1_elimination_has_no_parameter_conditions():
    net = ReactionNetwork.from_string("A + B -> C", species_names=["A", "B", "C"])
    an = LumpingAnalyzer(net)
    T = sp.Matrix([[1, 0, 0], [0, 1, 0]])
    res = an.find_critical_parameters(T)
    assert res["conditions"] == []


def test_construct_reduced_polynomial_system_for_simple_projection():
    net = ReactionNetwork.from_string("A + B -> C", species_names=["A", "B", "C"])
    an = LumpingAnalyzer(net)
    T = sp.Matrix([[1, 0, 0], [0, 1, 0]])
    red = an.construct_reduced_polynomial_system(T)

    y1, y2 = red["y_symbols"]
    G = red["G"]
    assert G.shape == (2, 1)

    k1 = net.rate_constants[0]

    # Should contain the mass action term -k1*y1*y2 in both equations.
    assert sp.simplify(G[0] + k1 * y1 * y2) == 0
    assert sp.simplify(G[1] + k1 * y1 * y2) == 0
