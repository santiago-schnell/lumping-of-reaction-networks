from __future__ import annotations

import sympy as sp

from .reaction import Reaction
from .network import ReactionNetwork


def three_species_linear_network() -> ReactionNetwork:
    """Example 4.2 from the paper: a three-species first-order network.

    Network:
        X1 <-> X2 <-> X3
    (no direct reaction between X1 and X3)

    Species order: [X1, X2, X3]
    Rate constants: k1, km1, k2, km2
    """
    k1, km1, k2, km2 = sp.symbols("k1 km1 k2 km2", positive=True)

    # X1 -> X2
    r1 = Reaction((1, 0, 0), (0, 1, 0), k1)
    # X2 -> X1
    r2 = Reaction((0, 1, 0), (1, 0, 0), km1)
    # X2 -> X3
    r3 = Reaction((0, 1, 0), (0, 0, 1), k2)
    # X3 -> X2
    r4 = Reaction((0, 0, 1), (0, 1, 0), km2)

    return ReactionNetwork(
        n_species=3,
        reactions=[r1, r2, r3, r4],
        species_names=["X1", "X2", "X3"],
    )


def michaelis_menten_network() -> ReactionNetwork:
    """Reversible Michaelis--Menten system (paper Example 4.3).

    Reaction scheme:
        S + E <-> C <-> E + P

    Species order: [S, E, C, P]
    Rate constants: k1, km1, k2, km2 (corresponding to k_{-1}, k_{-2} in the paper)
    """
    k1, km1, k2, km2 = sp.symbols("k1 km1 k2 km2", positive=True)

    # S + E -> C
    r1 = Reaction((1, 1, 0, 0), (0, 0, 1, 0), k1)
    # C -> S + E
    r2 = Reaction((0, 0, 1, 0), (1, 1, 0, 0), km1)

    # C -> E + P
    r3 = Reaction((0, 0, 1, 0), (0, 1, 0, 1), k2)
    # E + P -> C
    r4 = Reaction((0, 1, 0, 1), (0, 0, 1, 0), km2)

    return ReactionNetwork(
        n_species=4,
        reactions=[r1, r2, r3, r4],
        species_names=["S", "E", "C", "P"],
    )


def gpl_replication_network() -> ReactionNetwork:
    """Self-replication model (paper Section 6).

    Species mapping (paper notation):
        X1 = A, X2 = B, X3 = P, X4 = I_a, X5 = I_b, X6 = I

    Five reversible reactions:
        X1 + X3 <-> X4
        X2 + X4 <-> X6
        X2 + X3 <-> X5
        X1 + X5 <-> X6
        X6 <-> 2 X3

    Rate constants: k1, km1, ..., k5, km5.
    """
    k1, km1, k2, km2, k3, km3, k4, km4, k5, km5 = sp.symbols(
        "k1 km1 k2 km2 k3 km3 k4 km4 k5 km5", positive=True
    )

    # X1 + X3 <-> X4
    r1 = Reaction((1, 0, 1, 0, 0, 0), (0, 0, 0, 1, 0, 0), k1)
    r2 = Reaction((0, 0, 0, 1, 0, 0), (1, 0, 1, 0, 0, 0), km1)

    # X2 + X4 <-> X6
    r3 = Reaction((0, 1, 0, 1, 0, 0), (0, 0, 0, 0, 0, 1), k2)
    r4 = Reaction((0, 0, 0, 0, 0, 1), (0, 1, 0, 1, 0, 0), km2)

    # X2 + X3 <-> X5
    r5 = Reaction((0, 1, 1, 0, 0, 0), (0, 0, 0, 0, 1, 0), k3)
    r6 = Reaction((0, 0, 0, 0, 1, 0), (0, 1, 1, 0, 0, 0), km3)

    # X1 + X5 <-> X6
    r7 = Reaction((1, 0, 0, 0, 1, 0), (0, 0, 0, 0, 0, 1), k4)
    r8 = Reaction((0, 0, 0, 0, 0, 1), (1, 0, 0, 0, 1, 0), km4)

    # X6 <-> 2 X3
    r9 = Reaction((0, 0, 0, 0, 0, 1), (0, 0, 2, 0, 0, 0), k5)
    r10 = Reaction((0, 0, 2, 0, 0, 0), (0, 0, 0, 0, 0, 1), km5)

    return ReactionNetwork(
        n_species=6,
        reactions=[r1, r2, r3, r4, r5, r6, r7, r8, r9, r10],
        species_names=["A", "B", "P", "Ia", "Ib", "I"],
    )
