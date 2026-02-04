import sympy as sp

from lumping_analysis import ReactionNetwork, LumpingAnalyzer, michaelis_menten_network


def test_parser_explicit_reversible_rates():
    net = ReactionNetwork.from_string("A <->[k1][km1] B", species_names=["A", "B"])
    assert len(net.reactions) == 2
    assert str(net.reactions[0].rate) == "k1"
    assert str(net.reactions[1].rate) == "km1"


def test_proper_lumping_condition_for_michaelis_menten_symmetry():
    """Partition {S,P}|{E}|{C} yields k1 = km2 for the reversible MM network."""
    net = michaelis_menten_network()
    an = LumpingAnalyzer(net)

    # Species order: [S, E, C, P]
    res = an.proper_lumping_conditions([["S", "P"], ["E"], ["C"]])

    k1, km1, k2, km2 = net.rate_constants
    expected = sp.simplify(k1 - km2)

    assert any(
        sp.simplify(c - expected) == 0 or sp.simplify(c + expected) == 0
        for c in res["conditions"]
    )
