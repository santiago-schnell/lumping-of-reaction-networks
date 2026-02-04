from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence, Tuple

import sympy as sp


@dataclass(frozen=True)
class Reaction:
    """A single mass-action reaction.

    Parameters
    ----------
    reactants:
        Stoichiometric coefficients of the reactant complex (length n).
    products:
        Stoichiometric coefficients of the product complex (length n).
    rate:
        Symbol (or SymPy expression) for the rate constant.

    Notes
    -----
    For a reaction Y1 -> Y2 with reactant coefficients m_i and product
    coefficients r_i, mass action kinetics yields the term

        rate * prod_i x_i**m_i * (r - m)

    in the ODE system dx/dt.
    """

    reactants: Tuple[int, ...]
    products: Tuple[int, ...]
    rate: sp.Expr

    def __post_init__(self) -> None:
        if len(self.reactants) != len(self.products):
            raise ValueError("reactants and products must have the same length")
        if any(int(c) < 0 for c in self.reactants) or any(int(c) < 0 for c in self.products):
            raise ValueError("stoichiometric coefficients must be nonnegative integers")

    @property
    def n_species(self) -> int:
        return len(self.reactants)

    def reaction_vector(self) -> sp.Matrix:
        """Return v = products - reactants as an n×1 SymPy Matrix."""
        return sp.Matrix([int(p) - int(r) for r, p in zip(self.reactants, self.products)])

    def reactant_monomial(self, x: Sequence[sp.Symbol]) -> sp.Expr:
        """Return the mass-action monomial φ(x) = ∏ x_i^{m_i}."""
        if len(x) != self.n_species:
            raise ValueError("x must have length n_species")
        mon = sp.Integer(1)
        for xi, mi in zip(x, self.reactants):
            mi_int = int(mi)
            if mi_int:
                mon *= xi ** mi_int
        return sp.expand(mon)

    def contribution(self, x: Sequence[sp.Symbol]) -> sp.Matrix:
        """Return this reaction's contribution to the RHS F(x,k)."""
        phi = self.reactant_monomial(x)
        v = self.reaction_vector()
        return sp.Matrix(v) * sp.expand(self.rate * phi)

    def is_reactant(self, species_index: int) -> bool:
        """True iff the given species appears as a reactant (mi > 0)."""
        return int(self.reactants[species_index]) > 0

    def is_nonreactant(self, species_index: int) -> bool:
        """True iff the given species does not appear as a reactant (mi = 0)."""
        return int(self.reactants[species_index]) == 0

    @staticmethod
    def from_coeff_vectors(
        reactants: Iterable[int],
        products: Iterable[int],
        rate: sp.Expr,
    ) -> "Reaction":
        return Reaction(tuple(int(c) for c in reactants), tuple(int(c) for c in products), rate)
