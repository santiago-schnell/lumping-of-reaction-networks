import sympy as sp
from lumping_analysis import Reaction, ReactionNetwork, LumpingAnalyzer

def main():
    # Example 1:
    # X1 + X2 -> X3,  X4 + X5 -> X6
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

    res = an.generic_lumping_from_seed(["X1"])
    print("Layers:", res.layers)
    print("J:", res.J)
    print("Tagged reactions:", res.tagged_reactions)
    print("Untagged reactions:", res.untagged_reactions)
    print("T:")
    sp.pprint(res.T)

if __name__ == "__main__":
    main()
