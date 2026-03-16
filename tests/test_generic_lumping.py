import sympy as sp
from lumping_analysis.reaction import Reaction
from lumping_analysis.network import ReactionNetwork
from lumping_analysis.analyzer import LumpingAnalyzer

def test_generic_lumping_from_seed_example_1():
    """
    Tests the generic lumping closure procedure using Example 1:
    X1 + X2 -> X3
    X4 + X5 -> X6
    """
    k1, k2 = sp.symbols("k1 k2", positive=True)
    
    reactions = [
        Reaction((1, 1, 0, 0, 0, 0), (0, 0, 1, 0, 0, 0), k1),
        Reaction((0, 0, 0, 1, 1, 0), (0, 0, 0, 0, 0, 1), k2),
    ]
    
    net = ReactionNetwork(
        n_species=6,
        reactions=reactions,
        species_names=["X1", "X2", "X3", "X4", "X5", "X6"],
    )
    
    an = LumpingAnalyzer(net)
    
    # Test Seed J1 = {X1} (index 0)
    # Expected: The first reaction modifies X1, so its reactants (X1, X2) are added.
    # The second reaction is untouched.
    res_x1 = an.generic_lumping_from_seed(["X1"], remove_seeded_nonreactants=True)
    
    # Check that indices for X1 and X2 (0 and 1) are grouped
    assert set(res_x1.J) == {0, 1}, "Seed X1 should result in the set {X1, X2}"
    assert res_x1.T.shape == (2, 6)
    
    # Test Seed J1 = {X3} (index 2)
    # Expected: X3 is modified in the first reaction. Reactants X1, X2 are added.
    # X3 can be removed at the end since it's a non-reactant.
    res_x3 = an.generic_lumping_from_seed(["X3"], remove_seeded_nonreactants=True)
    
    # If remove_seeded_nonreactants is True, X3 should be stripped out, 
    # leaving only X1 and X2 to form the minimal rank projection.
    assert set(res_x3.J) == {0, 1}, "Seed X3 with removal should result in the set {X1, X2}"
    assert res_x3.T.shape == (2, 6)