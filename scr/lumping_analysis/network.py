from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import sympy as sp

from .reaction import Reaction
from .parser import ReactionParser


def _sanitize_symbol_name(name: str) -> str:
    # SymPy symbols may include many characters, but we keep a conservative subset
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_")
    if not name:
        return "x"
    cleaned = "".join(ch if ch in allowed else "_" for ch in name)
    if cleaned[0].isdigit():
        cleaned = "x_" + cleaned
    return cleaned


@dataclass
class ReactionNetwork:
    """A mass-action reaction network.

    Parameters
    ----------
    n_species:
        Number of species.
    reactions:
        List of `Reaction` objects.
    species_names:
        Optional list of length n with names (used for pretty symbols).

    Notes
    -----
    The ODE system is
        dx/dt = Σ_i k_i φ_i(x) v_i
    with mass action monomials φ_i and reaction vectors v_i.
    """

    n_species: int
    reactions: List[Reaction]
    species_names: Optional[List[str]] = None

    def __post_init__(self) -> None:
        if self.n_species <= 0:
            raise ValueError("n_species must be positive")
        if any(r.n_species != self.n_species for r in self.reactions):
            raise ValueError("all reactions must use the same n_species")
        if self.species_names is not None:
            if len(self.species_names) != self.n_species:
                raise ValueError("species_names must have length n_species")

        # Create SymPy symbols for species.
        if self.species_names is None:
            names = [f"x{i+1}" for i in range(self.n_species)]
        else:
            names = [_sanitize_symbol_name(s) for s in self.species_names]
        self._x = sp.Matrix(sp.symbols(" ".join(names), real=True))

    @property
    def x(self) -> sp.Matrix:
        """Species concentration symbols as an n×1 vector."""
        return self._x

    @property
    def x_symbols(self) -> Tuple[sp.Symbol, ...]:
        return tuple(self._x)

    @property
    def rate_constants(self) -> List[sp.Symbol]:
        """Unique rate symbols appearing in the reaction list (in order of appearance)."""
        seen = set()
        out: List[sp.Symbol] = []
        for r in self.reactions:
            if isinstance(r.rate, sp.Symbol):
                if r.rate not in seen:
                    seen.add(r.rate)
                    out.append(r.rate)
            else:
                # For expressions, collect symbols
                for sym in sorted(r.rate.free_symbols, key=lambda z: str(z)):
                    if sym not in seen:
                        seen.add(sym)
                        out.append(sym)
        return out

    def rhs(self) -> sp.Matrix:
        """Return the RHS F(x,k) as an n×1 SymPy Matrix."""
        F = sp.Matrix.zeros(self.n_species, 1)
        x = list(self.x_symbols)
        for r in self.reactions:
            F += r.contribution(x)
        return sp.simplify(F)

    def jacobian(self) -> sp.Matrix:
        """Return the Jacobian DF(x,k) as an n×n SymPy Matrix."""
        F = self.rhs()
        J = F.jacobian(self.x_symbols)
        return sp.simplify(J)

    def stoichiometric_matrix(self) -> sp.Matrix:
        """Return the stoichiometric matrix S with columns reaction vectors."""
        if not self.reactions:
            return sp.Matrix.zeros(self.n_species, 0)
        cols = [r.reaction_vector() for r in self.reactions]
        return sp.Matrix.hstack(*cols)

    def reactant_matrix(self) -> sp.Matrix:
        """Return the reactant stoichiometry matrix M with columns reactant complexes."""
        if not self.reactions:
            return sp.Matrix.zeros(self.n_species, 0)
        cols = [sp.Matrix(r.reactants) for r in self.reactions]
        return sp.Matrix.hstack(*cols)

    def stoichiometric_first_integrals(self, integer_basis: bool = True) -> List[sp.Matrix]:
        """Return a basis of stoichiometric first integrals.

        Each element is returned as a 1×n row vector μ^T such that μ^T S = 0.
        """
        S = self.stoichiometric_matrix()
        if S.cols == 0:
            # No reactions: every coordinate is an integral.
            basis = [sp.Matrix([[1 if i == j else 0 for j in range(self.n_species)]]) for i in range(self.n_species)]
            return basis

        # Left nullspace of S is right nullspace of S.T.
        null = S.T.nullspace()
        out: List[sp.Matrix] = []
        for v in null:
            # v is n×1 column; convert to row.
            row = sp.Matrix(v).T
            if integer_basis:
                row = _make_integer_row(row)
            out.append(row)
        return out

    def common_non_reactants(self) -> List[int]:
        """Return indices of species that never appear as reactants in any reaction."""
        idx: List[int] = []
        for i in range(self.n_species):
            if all(r.is_nonreactant(i) for r in self.reactions):
                idx.append(i)
        return idx

    def summary(self) -> str:
        """Human-readable summary."""
        lines = []
        lines.append(f"ReactionNetwork(n_species={self.n_species}, n_reactions={len(self.reactions)})")
        if self.species_names is not None:
            lines.append("Species: " + ", ".join(self.species_names))
        else:
            lines.append("Species symbols: " + ", ".join(str(s) for s in self.x_symbols))
        lines.append("Rate constants: " + ", ".join(str(k) for k in self.rate_constants))
        return "\n".join(lines)


    def to_latex(self) -> str:
        """Export the ODE system to LaTeX.

        Returns an ``align`` environment with equations of the form
            \\dot{x}_i = F_i(x,k).
        """
        F = self.rhs()
        xs = list(self.x_symbols)
        lines = []
        for i, xi in enumerate(xs):
            lhs = f"\\dot{{{sp.latex(xi)}}}"
            rhs = sp.latex(F[i, 0])
            lines.append(f"{lhs} &= {rhs}")
        body = " \\\\n".join(lines)
        return "\\begin{align}\n" + body + "\n\\end{align}"

    def reactions_to_latex(self) -> str:
        """Export directed reactions to LaTeX.

        Notes
        -----
        This prints each directed reaction separately. If your network is built from
        reversible reactions, you will see two lines per reversible pair.
        """

        def complex_to_str(coeffs):
            terms = []
            names = self.species_names or [str(s) for s in self.x_symbols]
            for name, c in zip(names, coeffs):
                c = int(c)
                if c == 0:
                    continue
                if c == 1:
                    terms.append(f"{name}")
                else:
                    terms.append(f"{c}{name}")
            return " + ".join(terms) if terms else "0"

        lines = []
        for r in self.reactions:
            lhs = complex_to_str(r.reactants)
            rhs = complex_to_str(r.products)
            k = sp.latex(r.rate)
            lines.append(f"{lhs} \\xrightarrow{{{k}}} {rhs}")

        body = " \\\\n".join(lines)
        return "\\begin{align}\n" + body + "\n\\end{align}"

    # -----------------------------
    # Constructors
    # -----------------------------

    @classmethod
    def from_string(
        cls,
        text: str,
        species_names: Optional[Sequence[str]] = None,
        rate_prefix: str = "k",
    ) -> "ReactionNetwork":
        """Parse a reaction network from a multi-line string.

        Parameters
        ----------
        text:
            Reaction lines separated by newlines or semicolons.
            Supported arrows: "->"/"=>" (irreversible) and "<->"/"<=>" (reversible).

            Optional rate constants can be provided in brackets immediately after the arrow, e.g.:
              - "A + B ->[k1] C"
              - "A + B <->[k1][km1] C"   (or "<->[k1, km1]")
        species_names:
            Optional explicit ordering of species.
        rate_prefix:
            Prefix used when auto-generating rate constants.

        Returns
        -------
        ReactionNetwork
        """
        parser = ReactionParser(rate_prefix=rate_prefix)
        return parser.parse_network(text=text, species_names=species_names)


def _make_integer_row(row: sp.Matrix) -> sp.Matrix:
    """Scale a rational row vector to a primitive integer row vector."""
    if row.shape[0] != 1:
        raise ValueError("row must be 1×n")

    # Clear denominators.
    dens = []
    nums = []
    for entry in row.tolist()[0]:
        num, den = sp.fraction(sp.nsimplify(entry))
        nums.append(num)
        dens.append(den)

    lcm = sp.ilcm(*[int(d) for d in dens]) if dens else 1
    scaled = [sp.expand(num * (lcm // int(den))) for num, den in zip(nums, dens)]

    # Make primitive by dividing gcd.
    ints = [int(sp.Integer(s)) for s in scaled]
    if all(v == 0 for v in ints):
        return row
    g = abs(ints[0])
    for v in ints[1:]:
        g = sp.igcd(g, abs(v))
    g = int(g) if g != 0 else 1
    prim = [sp.Integer(v // g) for v in ints]

    # Canonical sign: make first nonzero entry positive.
    for v in prim:
        if v != 0:
            if v < 0:
                prim = [-vv for vv in prim]
            break

    return sp.Matrix([prim])
