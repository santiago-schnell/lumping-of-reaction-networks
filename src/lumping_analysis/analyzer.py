from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp
from sympy.solvers.solveset import NonlinearError

from .network import ReactionNetwork


def _matrix_from_nullspace(nullspace: List[sp.Matrix], n: int) -> sp.Matrix:
    """Stack a list of n×1 vectors into an n×m matrix."""
    if not nullspace:
        return sp.Matrix.zeros(n, 0)
    cols = [sp.Matrix(v) for v in nullspace]
    return sp.Matrix.hstack(*cols)


def _polynomial_coeff_conditions(expr: sp.Expr, x_symbols: Sequence[sp.Symbol]) -> List[sp.Expr]:
    """Return coefficient conditions for expr == 0 for all x.

    We treat expr as a rational expression in x, clear denominators, then extract
    coefficients of the resulting polynomial in x.
    """
    expr = sp.together(sp.expand(expr))
    num, den = sp.fraction(expr)
    num = sp.expand(num)

    if num == 0:
        return []

    try:
        poly = sp.Poly(num, *x_symbols)
    except sp.PolynomialError:
        # As a fallback, force polynomial expansion by treating non-x symbols as coefficients.
        poly = sp.Poly(sp.expand(num), *x_symbols, domain="EX")

    return [sp.simplify(c) for c in poly.coeffs()]


def _unique_simplified(exprs: Iterable[sp.Expr]) -> List[sp.Expr]:
    """De-duplicate expressions after simplify, identifying c and -c as equivalent.

    Many coefficient-extraction workflows generate redundant conditions that differ
    only by an overall nonzero scalar factor (often just a sign). For purposes of
    describing the algebraic variety, c=0 and -c=0 are the same.
    """
    out: List[sp.Expr] = []
    seen = set()
    for e in exprs:
        ee = sp.simplify(e)
        if ee == 0:
            continue
        k1 = sp.srepr(ee)
        k2 = sp.srepr(-ee)
        if k1 in seen or k2 in seen:
            continue
        seen.add(k1)
        out.append(ee)
    return out


def _analyze_homogeneous_linear_system(A: sp.Matrix, vars: Sequence[sp.Symbol]) -> Dict[str, Any]:
    """Analyze the homogeneous linear system A*vars = 0.

    This is a small reporting/diagnostic helper for the common situation
    (discussed in the paper, Section 4) where the Jacobian invariance
    condition yields a *linear* homogeneous system in the rate constants.

    The analysis is purely algebraic (symbolic) and is intended as a *diagnostic*
    and reporting aid rather than a complete algebraic-geometry solver.

    Returns
    -------
    dict with keys:
      - 'rref': reduced row echelon form of A
      - 'pivots': pivot column indices
      - 'relations': list of linear expressions r_i(vars) == 0 implied by the rref
      - 'forced_zero': subset of vars that are forced to be identically zero
    """
    if A.cols != len(vars):
        raise ValueError("A.cols must match len(vars)")

    rref, pivots = A.rref()

    relations: List[sp.Expr] = []
    forced_zero: List[sp.Symbol] = []
    for i in range(rref.rows):
        row = list(rref.row(i))
        if all(sp.simplify(c) == 0 for c in row):
            continue
        expr = sp.simplify(sum(row[j] * vars[j] for j in range(len(vars))))
        if expr != 0:
            relations.append(expr)

        # Detect rows of the form k_j = 0.
        nz = [j for j, c in enumerate(row) if sp.simplify(c) != 0]
        if len(nz) == 1:
            forced_zero.append(vars[nz[0]])

    # De-duplicate forced_zero (symbolic rref sometimes repeats rows).
    forced_zero = list(dict.fromkeys(forced_zero))

    return {
        "rref": rref,
        "pivots": pivots,
        "relations": relations,
        "forced_zero": forced_zero,
    }


def _partitions(seq: List[int]) -> Iterator[List[List[int]]]:
    """Generate all set partitions of a list of ints.

    Notes
    -----
    - The order of blocks in each yielded partition is canonical for this generator.
    - Each block is a list (not a set) whose elements appear in the same order as `seq`.
    """
    if not seq:
        yield []
        return

    first, rest = seq[0], seq[1:]
    for part in _partitions(rest):
        # Put `first` into an existing block.
        for i in range(len(part)):
            new_part = [blk[:] for blk in part]
            new_part[i] = [first] + new_part[i]
            yield new_part
        # Or start a new block.
        yield [[first]] + [blk[:] for blk in part]


def _canonical_blocks(blocks: Sequence[Sequence[int]]) -> Tuple[Tuple[int, ...], ...]:
    """Canonicalize a partition given as blocks of indices."""
    blks = [tuple(sorted(int(i) for i in blk)) for blk in blocks]
    # Sort blocks by (size, lexicographic)
    blks.sort(key=lambda b: (len(b), b))
    return tuple(blks)


def _blocks_from_names_or_indices(
    blocks: Sequence[Sequence[Union[int, str]]],
    *,
    species_names: Optional[Sequence[str]],
    n_species: int,
) -> List[List[int]]:
    """Convert blocks expressed as indices or species names into index blocks."""
    if not blocks:
        raise ValueError("blocks must be non-empty")

    name_to_idx: Dict[str, int] = {}
    if species_names is not None:
        name_to_idx = {str(nm): i for i, nm in enumerate(species_names)}

    out: List[List[int]] = []
    for blk in blocks:
        if not blk:
            raise ValueError("blocks may not contain empty blocks")
        idxs: List[int] = []
        for item in blk:
            if isinstance(item, int):
                idx = int(item)
            else:
                if not name_to_idx:
                    raise ValueError(
                        "Blocks contain species names, but this network has no species_names. "
                        "Provide indices instead."
                    )
                if str(item) not in name_to_idx:
                    raise ValueError(f"Unknown species name '{item}' in blocks")
                idx = name_to_idx[str(item)]
            if not (0 <= idx < n_species):
                raise ValueError(f"Species index out of range: {idx}")
            idxs.append(idx)
        out.append(sorted(set(idxs)))

    # Validate disjoint union covers all species.
    flat = [i for blk in out for i in blk]
    if len(flat) != len(set(flat)):
        raise ValueError("blocks must be disjoint")
    if set(flat) != set(range(n_species)):
        raise ValueError("blocks must form a partition of {0,...,n-1}")

    return out


@dataclass
class LumpingAnalyzer:
    """Main analysis class.

    Parameters
    ----------
    network:
        A `ReactionNetwork`.
    """

    network: ReactionNetwork

    def find_generic_lumpings(self) -> Dict[str, Any]:
        """Compute the two generic lumping families from the paper.

        Returns
        -------
        dict with keys:
          - 'common_non_reactants': list of species names (or symbols)
          - 'stoichiometric_integrals': list of 1×n SymPy row vectors
        """
        nonreact_idx = self.network.common_non_reactants()
        if self.network.species_names is not None:
            nonreact_names = [self.network.species_names[i] for i in nonreact_idx]
        else:
            nonreact_names = [str(self.network.x_symbols[i]) for i in nonreact_idx]

        integrals = self.network.stoichiometric_first_integrals(integer_basis=True)

        return {
            "common_non_reactants": nonreact_names,
            "common_non_reactant_indices": nonreact_idx,
            "stoichiometric_integrals": integrals,
        }

    def kernel_basis(self, T: sp.Matrix) -> sp.Matrix:
        """Return a basis matrix B for ker(T) as an n×(n-rank(T)) matrix."""
        n = self.network.n_species
        if T.cols != n:
            raise ValueError(f"T must have {n} columns")
        null = T.nullspace()
        return _matrix_from_nullspace(null, n)

    def build_constrained_lumping_matrix(
        self,
        *,
        prescribed_rows: Optional[Sequence[sp.Matrix]] = None,
        n_free_rows: int = 1,
        free_param_prefix: str = "p",
    ) -> Tuple[sp.Matrix, List[sp.Symbol]]:
        """Build a lumping matrix with prescribed rows and free parameters.

        This is a convenience constructor for *constrained lumping* scenarios
        (paper Remark 10): you specify a set of linear observables that must be
        preserved (rows of T), then add additional free rows to search for
        critical-parameter reductions.

        Parameters
        ----------
        prescribed_rows:
            List of 1×n row vectors (or n×1 vectors, which will be transposed)
            to include as fixed rows of T.
        n_free_rows:
            Number of additional rows populated by distinct free parameters.
        free_param_prefix:
            Prefix used when creating free parameter symbols.

        Returns
        -------
        (T, free_params)
            T is an (e×n) matrix with e = len(prescribed_rows) + n_free_rows.
            free_params is the flat list of newly created SymPy Symbols.

        Examples
        --------
        >>> integrals = net.stoichiometric_first_integrals(integer_basis=True)
        >>> T, params = analyzer.build_constrained_lumping_matrix(
        ...     prescribed_rows=integrals, n_free_rows=1
        ... )
        """
        n = self.network.n_species
        prescribed_rows = list(prescribed_rows) if prescribed_rows is not None else []

        fixed_rows: List[sp.Matrix] = []
        for row in prescribed_rows:
            r = sp.Matrix(row)
            if r.shape == (n, 1):
                r = r.T
            if r.shape != (1, n):
                raise ValueError(f"Each prescribed row must be 1×{n} (or {n}×1); got {r.shape}")
            fixed_rows.append(r)

        if n_free_rows < 0:
            raise ValueError("n_free_rows must be nonnegative")

        free_params: List[sp.Symbol] = []
        free_rows: List[sp.Matrix] = []
        for i in range(int(n_free_rows)):
            row_entries: List[sp.Symbol] = []
            for j in range(n):
                p = sp.Symbol(f"{free_param_prefix}_{i+1}{j+1}")
                free_params.append(p)
                row_entries.append(p)
            free_rows.append(sp.Matrix([row_entries]))

        all_rows = fixed_rows + free_rows
        if not all_rows:
            raise ValueError("At least one row is required to build T")

        T = sp.Matrix.vstack(*all_rows)
        return T, free_params

    def critical_conditions(self, T: sp.Matrix, kernel_basis: Optional[sp.Matrix] = None) -> Dict[str, Any]:
        """Compute polynomial conditions for T to be an exact lumping map.

        This implements the Jacobian invariance condition in the form
            T * DF(x,k) * B = 0  (for all x)
        where columns of B span ker(T).

        Returns
        -------
        dict with keys:
          - 'kernel_basis': B
          - 'TJB': T*J*B
          - 'conditions': list of polynomial conditions (in k and any free params)
        """
        n = self.network.n_species
        if T.cols != n:
            raise ValueError(f"T must have {n} columns")

        J = self.network.jacobian()
        B = kernel_basis if kernel_basis is not None else self.kernel_basis(T)
        if B.cols == 0:
            raise ValueError("ker(T) is trivial; no dimension reduction")

        TJB = sp.simplify(T * J * B)

        # Extract coefficient conditions in x.
        conds: List[sp.Expr] = []
        xsyms = self.network.x_symbols
        for entry in list(TJB):
            conds.extend(_polynomial_coeff_conditions(entry, xsyms))

        conds = _unique_simplified(conds)
        conds = [sp.factor(c) for c in conds]

        return {
            "kernel_basis": B,
            "TJB": TJB,
            "conditions": conds,
        }

    def find_critical_parameters(
        self,
        T: sp.Matrix,
        *,
        kernel_basis: Optional[sp.Matrix] = None,
        solve_for_rate_constants: bool = True,
        rate_constants: Optional[Sequence[sp.Symbol]] = None,
    ) -> Dict[str, Any]:
        """Compute critical-parameter conditions for a given candidate lumping matrix T.

        Parameters
        ----------
        T:
            Lumping matrix (e×n) with full row rank.
        kernel_basis:
            Optional precomputed basis matrix B for ker(T).
        solve_for_rate_constants:
            If True, try to solve the resulting equations as a *linear* system in the
            rate constants.
        rate_constants:
            Optionally override the list of rate constant symbols.

        Returns
        -------
        dict
            A structured result including conditions and (when possible) a parametric
            rate-constant solution.
        """
        info = self.critical_conditions(T, kernel_basis=kernel_basis)
        conds = info["conditions"]

        solns: Dict[str, Any] = {}
        if solve_for_rate_constants:
            k_syms = list(rate_constants) if rate_constants is not None else self.network.rate_constants
            if k_syms:
                try:
                    # Pass expressions directly (interpreted as == 0) to avoid
                    # SymPy simplifying Eq(0,0) into boolean True/False.
                    A, b = sp.linear_eq_to_matrix(conds, k_syms)
                    # Solve A*k = b (expected b=0).
                    sol_set = sp.linsolve((A, b), *k_syms)
                    solns["rate_constant_solution"] = sol_set
                    solns["linear_system_matrix"] = A
                    solns["linear_system_rhs"] = b
                    # Diagnostics to make the induced linear constraints easy to report.
                    if b == sp.zeros(A.rows, 1):
                        solns["linear_system_analysis"] = _analyze_homogeneous_linear_system(A, k_syms)
                except (NonlinearError, ValueError, sp.PolynomialError, TypeError):
                    solns["rate_constant_solution"] = None
            else:
                solns["rate_constant_solution"] = None

        return {
            "T": T,
            "kernel_basis": info["kernel_basis"],
            "TJB": info["TJB"],
            "conditions": conds,
            "solutions": solns,
        }



    # ---------------------------------------------------------------------
    # Row-echelon ansatz (general linear lumpings) and CAS-oriented workflows
    # ---------------------------------------------------------------------

    def row_echelon_ansatz(
        self,
        e: int,
        *,
        pivot_cols: Optional[Sequence[int]] = None,
        free_param_prefix: str = "t",
        flat_numbering: bool = True,
    ) -> Dict[str, Any]:
        """Construct a row-echelon ansatz for a general linear lumping map.

        This mirrors the paper's discussion: for a desired reduced dimension `e`
        (with `1 <= e < n`), choose a pivot set `I` of size `e`. Up to a permutation
        of columns, a full-row-rank lumping matrix can be assumed to have the form

            T_perm = [ I_e  |  Tbar ],

        where `Tbar` is an `e × (n-e)` matrix of free parameters. A convenient
        polynomial kernel basis is then

            B_perm = [ -Tbar ]
                     [  I    ].

        When `pivot_cols` is not the first `e` columns, we permute columns back to
        the original species ordering, producing matrices (T, B) satisfying T*B = 0.

        Parameters
        ----------
        e:
            Number of lumped variables (rows of T).
        pivot_cols:
            Indices (0-based) of the pivot columns. If None, uses [0, 1, ..., e-1].
        free_param_prefix:
            Prefix for newly created free parameters (default: 't').
        flat_numbering:
            If True, free parameters are named t1, t2, ... (as in the attached notebook).
            If False, they are named t_ij.

        Returns
        -------
        dict with keys:
          - 'T': e×n row-echelon ansatz matrix in the original column order
          - 'B': n×(n-e) polynomial kernel-basis matrix (satisfying T*B = 0)
          - 'free_params': list of free parameter Symbols used in Tbar
          - 'pivot_cols', 'nonpivot_cols', 'column_order'
          - 'permutation_matrix': n×n matrix P such that (T*P) = T_perm
        """
        n = self.network.n_species
        if not (1 <= int(e) < n):
            raise ValueError(f"e must satisfy 1 <= e < n={n}; got {e}")

        if pivot_cols is None:
            piv = list(range(int(e)))
        else:
            piv = [int(i) for i in pivot_cols]

        if len(piv) != int(e) or len(set(piv)) != int(e):
            raise ValueError(f"pivot_cols must be a list of {e} distinct indices")
        if any(i < 0 or i >= n for i in piv):
            raise ValueError(f"pivot_cols entries must be in [0, {n-1}]")

        nonpiv = [j for j in range(n) if j not in piv]
        order = list(piv) + nonpiv  # pivot columns first

        # Permutation matrix P such that for any matrix M, M*P has columns reordered by `order`.
        P = sp.Matrix.zeros(n, n)
        for new_j, old_j in enumerate(order):
            P[old_j, new_j] = 1

        # Free parameter block Tbar (e × (n-e)).
        nfree = n - int(e)
        free_params: List[sp.Symbol] = []
        if nfree > 0:
            if flat_numbering:
                cnt = 1
                Tbar = sp.Matrix.zeros(int(e), nfree)
                for i in range(int(e)):
                    for j in range(nfree):
                        t = sp.Symbol(f"{free_param_prefix}{cnt}")
                        free_params.append(t)
                        Tbar[i, j] = t
                        cnt += 1
            else:
                Tbar = sp.Matrix.zeros(int(e), nfree)
                for i in range(int(e)):
                    for j in range(nfree):
                        t = sp.Symbol(f"{free_param_prefix}_{i+1}{j+1}")
                        free_params.append(t)
                        Tbar[i, j] = t
        else:
            Tbar = sp.Matrix.zeros(int(e), 0)

        T_perm = sp.Matrix.hstack(sp.eye(int(e)), Tbar)  # e×n
        B_perm = sp.Matrix.vstack(-Tbar, sp.eye(nfree))  # n×(n-e)

        # Undo the column permutation: T = T_perm * P^{-1} = T_perm * P.T
        T = sp.simplify(T_perm * P.T)
        # Transform kernel basis to original variable ordering: B = P * B_perm
        B = sp.simplify(P * B_perm)

        return {
            "T": T,
            "B": B,
            "free_params": free_params,
            "pivot_cols": piv,
            "nonpivot_cols": nonpiv,
            "column_order": order,
            "permutation_matrix": P,
        }

    def complete_to_invertible(
        self,
        T: sp.Matrix,
        *,
        extra_rows: Optional[Sequence[sp.Matrix]] = None,
        strategy: str = "identity",
    ) -> sp.Matrix:
        """Complete an e×n matrix T to an n×n invertible matrix T*.

        This supports the coordinate-change construction discussed in the paper
        ("lumping-adapted" system): given a lumping matrix T, append (n-e) rows
        to obtain an invertible matrix T*, then define y = T* x and transform the
        ODE as y' = T* F(T*^{-1} y, k).

        By default (`strategy='identity'`), we greedily append rows from the
        identity matrix.

        Parameters
        ----------
        T:
            e×n matrix (usually full row rank).
        extra_rows:
            Optional explicit rows to append (each 1×n or n×1).
        strategy:
            Currently only 'identity' is implemented.

        Returns
        -------
        T_star:
            n×n matrix with top e rows equal to T.
        """
        n = self.network.n_species
        Tm = sp.Matrix(T)
        if Tm.cols != n:
            raise ValueError(f"T must have {n} columns; got {Tm.cols}")

        e = Tm.rows
        if e > n:
            raise ValueError("T has more rows than columns; cannot complete to square invertible matrix")
        if e == n:
            # Already square; check invertible (rank test).
            if Tm.rank() != n:
                raise ValueError("Provided square T is singular; cannot invert")
            return Tm

        base_rank = Tm.rank()

        rows_to_try: List[sp.Matrix] = []
        if extra_rows is not None:
            for r in extra_rows:
                rr = sp.Matrix(r)
                if rr.shape == (n, 1):
                    rr = rr.T
                if rr.shape != (1, n):
                    raise ValueError(f"Each extra row must be 1×{n} (or {n}×1); got {rr.shape}")
                rows_to_try.append(rr)
        else:
            if strategy != "identity":
                raise ValueError(f"Unknown completion strategy '{strategy}'")
            I = sp.eye(n)
            rows_to_try = [I.row(i) for i in range(n)]

        T_star = sp.Matrix(Tm)
        rnk = base_rank
        for rr in rows_to_try:
            if T_star.rows >= n:
                break
            cand = sp.Matrix.vstack(T_star, rr)
            cand_rank = cand.rank()
            if cand_rank > rnk:
                T_star = cand
                rnk = cand_rank

        if T_star.rows != n or T_star.rank() != n:
            raise ValueError(
                "Failed to complete T to an invertible n×n matrix. "
                "Try providing `extra_rows=` explicitly (or choose a different strategy)."
            )

        return T_star

    def lumping_adapted_system(
        self,
        T: sp.Matrix,
        *,
        completion_rows: Optional[Sequence[sp.Matrix]] = None,
        completion_strategy: str = "identity",
        y_prefix: str = "y",
        simplify: bool = True,
    ) -> Dict[str, Any]:
        """Compute a "lumping-adapted" coordinate representation of the ODE.

        Given x' = F(x,k) and a candidate lumping matrix T (e×n), form an invertible
        completion T* (n×n) by appending rows, define the new coordinates

            y = T* x,

        and compute the transformed vector field

            y' = H(y,k) = T* F(T*^{-1} y, k).

        The first e components of y are exactly the lumped variables y_1..y_e = T x
        (because the top e rows of T* equal T).

        This is the construction described in Sebastian's note (and referenced in the
        paper around Remark 5 / Example 7).

        Returns
        -------
        dict with keys:
          - 'T_star', 'T_star_inv'
          - 'y_symbols': tuple of length n
          - 'x_in_terms_of_y': n×1 vector
          - 'H': n×1 transformed RHS (in y variables)
          - 'H_lumped': first e entries of H
          - 'lumped_depends_on_extra': bool (whether H_lumped depends on y_{e+1..n})
        """
        n = self.network.n_species
        Tm = sp.Matrix(T)
        if Tm.cols != n:
            raise ValueError(f"T must have {n} columns; got {Tm.cols}")

        e = Tm.rows
        if not (1 <= e <= n):
            raise ValueError("T must have at least one row and at most n rows")

        T_star = self.complete_to_invertible(
            Tm, extra_rows=completion_rows, strategy=completion_strategy
        )

        # Symbols and substitutions.
        y_syms = sp.symbols(" ".join([f"{y_prefix}{i+1}" for i in range(n)]), real=True)
        y_vec = sp.Matrix(y_syms)

        T_star_inv = sp.simplify(T_star.inv())
        x_vec = sp.simplify(T_star_inv * y_vec)

        x_syms = self.network.x_symbols
        subs = {x_syms[i]: x_vec[i, 0] for i in range(n)}

        F = self.network.rhs()
        F_y = sp.Matrix([sp.together(F[i, 0].subs(subs)) for i in range(n)])
        H = sp.Matrix([sp.together((T_star * F_y)[i, 0]) for i in range(n)])

        if simplify:
            H = sp.Matrix([sp.simplify(h) for h in list(H)])
            x_vec = sp.Matrix([sp.simplify(xx) for xx in list(x_vec)])

        H_lumped = sp.Matrix([H[i, 0] for i in range(e)])
        extra_y = set(y_syms[e:])
        lumped_depends_on_extra = any(bool(set(expr.free_symbols) & extra_y) for expr in list(H_lumped))

        return {
            "T": Tm,
            "T_star": T_star,
            "T_star_inv": T_star_inv,
            "y_symbols": tuple(y_syms),
            "x_in_terms_of_y": x_vec,
            "H": H,
            "H_lumped": H_lumped,
            "lumped_depends_on_extra": lumped_depends_on_extra,
        }

    def critical_ideal(
        self,
        T: sp.Matrix,
        *,
        kernel_basis: Optional[sp.Matrix] = None,
        variables: Optional[Sequence[sp.Symbol]] = None,
    ) -> Any:
        """Package the critical conditions for T as a `SingularIdeal` for export.

        This is a convenience wrapper: it computes the critical-parameter
        conditions and constructs a `SingularIdeal` whose generators are those
        polynomial conditions.

        The returned object lives in :mod:`lumping_analysis.singular` so that
        ideal computations can be delegated to Singular.
        """
        from .singular import SingularIdeal  # local import to avoid hard dependency patterns

        info = self.critical_conditions(T, kernel_basis=kernel_basis)
        gens = info["conditions"]

        if variables is None:
            # Default ring variables = all non-state symbols appearing in the generators.
            xset = set(self.network.x_symbols)
            syms: set = set()
            for g in gens:
                syms |= {s for s in g.free_symbols if s not in xset}
            variables = sorted(syms, key=lambda s: str(s))

        return SingularIdeal.from_generators(gens, variables=variables)

    # ---------------------------------------------------------------------
    # Proper lumping (paper Section 5)
    # ---------------------------------------------------------------------

    def proper_lumping_matrix(
        self,
        blocks: Sequence[Sequence[Union[int, str]]],
        *,
        weights: Optional[Sequence[sp.Expr]] = None,
    ) -> sp.Matrix:
        r"""Build a **proper lumping** matrix T from a partition of species.

        The unweighted proper lumping map is
            y_p = \sum_{j \in I_p} x_j,
        corresponding to a 0/1 matrix T with exactly one 1 in each column.

        Parameters
        ----------
        blocks:
            A partition of species into blocks. Blocks may be given by indices
            (0-based) or by species names (if `network.species_names` is set).
        weights:
            Optional per-species weights \gamma_j (length n). When provided,
            entries of T become \gamma_j instead of 1.

        Notes
        -----
        The column-sum criterion from Proposition 5.1 (paper) applies to the
        **unweighted** case. For weighted proper lumping, you can still use the
        general condition `find_critical_parameters(T)`.
        """
        n = self.network.n_species
        idx_blocks = _blocks_from_names_or_indices(
            blocks, species_names=self.network.species_names, n_species=n
        )
        r = len(idx_blocks)
        if weights is not None:
            if len(weights) != n:
                raise ValueError(f"weights must have length n={n}")
            w = list(weights)
        else:
            w = [sp.Integer(1)] * n

        T = sp.Matrix.zeros(r, n)
        for p, blk in enumerate(idx_blocks):
            for j in blk:
                T[p, j] = w[j]
        return T

    def proper_lumping_conditions(
        self,
        blocks: Sequence[Sequence[Union[int, str]]],
        *,
        solve_for_rate_constants: bool = True,
        rate_constants: Optional[Sequence[sp.Symbol]] = None,
    ) -> Dict[str, Any]:
        """Compute critical-parameter conditions for an **unweighted** proper lumping.

        Implements the column-sum criterion from Proposition 5.1 (paper):
        for each block pair (p,q), all column sums of the Jacobian submatrix
        M_{pq} must be equal.

        Returns
        -------
        dict with keys:
          - 'blocks' and 'blocks_indices'
          - 'T': the proper lumping matrix
          - 'conditions': algebraic conditions (polynomials)
          - 'solutions': optional linear solution space for rate constants
        """
        n = self.network.n_species
        idx_blocks = _blocks_from_names_or_indices(
            blocks, species_names=self.network.species_names, n_species=n
        )
        can_blocks = _canonical_blocks(idx_blocks)

        J = self.network.jacobian()
        xsyms = self.network.x_symbols

        conds: List[sp.Expr] = []
        for Ip in can_blocks:
            for Iq in can_blocks:
                if len(Iq) <= 1:
                    continue

                base = Iq[0]
                base_sum = sp.simplify(sum(J[i, base] for i in Ip))

                for j in Iq[1:]:
                    col_sum = sp.simplify(sum(J[i, j] for i in Ip))
                    diff = sp.simplify(col_sum - base_sum)
                    conds.extend(_polynomial_coeff_conditions(diff, xsyms))

        conds = _unique_simplified(conds)
        conds = [sp.factor(c) for c in conds]

        T = self.proper_lumping_matrix(can_blocks)

        solns: Dict[str, Any] = {}
        if solve_for_rate_constants:
            k_syms = list(rate_constants) if rate_constants is not None else self.network.rate_constants
            if k_syms:
                try:
                    # Pass expressions directly (interpreted as == 0) to avoid
                    # SymPy simplifying Eq(0,0) into boolean True/False.
                    A, b = sp.linear_eq_to_matrix(conds, k_syms)
                    sol_set = sp.linsolve((A, b), *k_syms)
                    solns["rate_constant_solution"] = sol_set
                    solns["linear_system_matrix"] = A
                    solns["linear_system_rhs"] = b
                    if b == sp.zeros(A.rows, 1):
                        solns["linear_system_analysis"] = _analyze_homogeneous_linear_system(A, k_syms)
                except (NonlinearError, ValueError, sp.PolynomialError, TypeError):
                    solns["rate_constant_solution"] = None
            else:
                solns["rate_constant_solution"] = None

        return {
            "blocks_indices": [list(b) for b in can_blocks],
            "blocks": (
                [[self.network.species_names[i] for i in b] for b in can_blocks]
                if self.network.species_names is not None
                else [list(b) for b in can_blocks]
            ),
            "T": T,
            "conditions": conds,
            "solutions": solns,
        }

    def find_proper_lumpings(
        self,
        *,
        n_blocks: Optional[int] = None,
        max_partitions: int = 500,
        solve_for_rate_constants: bool = True,
    ) -> List[Dict[str, Any]]:
        """Search over partitions to find proper lumping candidates.

        Parameters
        ----------
        n_blocks:
            If provided, restrict to partitions with exactly this number of
            blocks (i.e., reduced dimension e = n_blocks).
        max_partitions:
            Safety cap on the number of partitions examined.
        solve_for_rate_constants:
            If True, return a (possibly parametric) linear solution set for the
            rate constants whenever the conditions are detected to be linear.

        Notes
        -----
        The number of partitions grows rapidly with n (Bell numbers). For n>9,
        exhaustive enumeration becomes expensive.
        """
        n = self.network.n_species
        idxs = list(range(n))

        results: List[Dict[str, Any]] = []
        examined = 0
        for part in _partitions(idxs):
            if n_blocks is not None and len(part) != n_blocks:
                continue
            if len(part) == n:
                # No dimension reduction.
                continue

            res = self.proper_lumping_conditions(part, solve_for_rate_constants=solve_for_rate_constants)
            results.append(res)
            examined += 1
            if examined >= int(max_partitions):
                break

        return results

    # ---------------------------------------------------------------------
    # Reduced system construction (small systems)
    # ---------------------------------------------------------------------

    def construct_reduced_polynomial_system(
        self,
        T: sp.Matrix,
        *,
        degree: Optional[int] = None,
        parameter_subs: Optional[Dict[sp.Symbol, Any]] = None,
    ) -> Dict[str, Any]:
        """Attempt to construct a polynomial reduced system y' = G(y).

        This follows a Li–Rabitz style approach:

        - Let y = T x.
        - Compute y' = T F(x,k).
        - Assume G is a polynomial in y (degree <= `degree`).
        - Solve for coefficients of G by equating polynomials in x:
              G(Tx) == T F(x,k).

        Notes
        -----
        This is intended for **small systems** (small e and moderate polynomial degree).

        Returns
        -------
        dict with keys:
          - 'y_symbols': tuple of y symbols
          - 'G': e×1 SymPy Matrix representing G(y)
          - 'degree': used degree
        """
        n = self.network.n_species
        if T.cols != n:
            raise ValueError(f"T must have {n} columns")

        e = T.rows
        x = sp.Matrix(self.network.x_symbols)

        F = self.network.rhs()
        if parameter_subs is not None:
            F = F.subs(parameter_subs)

        y_expr = sp.simplify(T * x)
        y_syms = sp.symbols(" ".join([f"y{i+1}" for i in range(e)]), real=True)

        ydot = sp.simplify(T * F)

        # Choose polynomial degree in y.
        if degree is None:
            # Use maximal total degree of the ydot entries in x.
            degs = []
            for comp in list(ydot):
                try:
                    degs.append(sp.Poly(comp, *self.network.x_symbols).total_degree())
                except sp.PolynomialError:
                    degs.append(1)
            degree = max(degs) if degs else 1

        # Monomial basis in y up to degree.
        mon_set = sp.itermonomials(y_syms, degree)
        monomials = sorted(mon_set, key=lambda m: (sp.total_degree(m), sp.default_sort_key(m)))

        # Unknown coefficients for each component.
        coeffs: List[sp.Symbol] = []
        G_entries: List[sp.Expr] = []
        equations: List[sp.Expr] = []

        for i in range(e):
            ci = sp.symbols(" ".join([f"c_{i+1}_{j+1}" for j in range(len(monomials))]), real=True)
            coeffs.extend(ci)
            g_i = sum(c * m for c, m in zip(ci, monomials))
            G_entries.append(g_i)

            g_sub = sp.expand(g_i.subs({y_syms[j]: y_expr[j, 0] for j in range(e)}))
            target = sp.expand(ydot[i, 0])
            diff = sp.expand(g_sub - target)

            # Coefficient equations in x.
            poly = sp.Poly(diff, *self.network.x_symbols, domain="EX")
            equations.extend([sp.Eq(c, 0) for c in poly.coeffs()])

        # Solve linear system for coefficients.
        A, b = sp.linear_eq_to_matrix(equations, coeffs)
        sol = sp.linsolve((A, b), *coeffs)
        if not sol:
            raise ValueError("No polynomial reduced system found (system inconsistent)")

        # Pick one solution tuple.
        sol_tuple = next(iter(sol))
        subs = {c: v for c, v in zip(coeffs, sol_tuple)}

        G = sp.Matrix([sp.simplify(g.subs(subs)) for g in G_entries])

        return {
            "y_symbols": tuple(y_syms),
            "G": G,
            "degree": degree,
            "monomials": monomials,
            "coefficient_solution": subs,
        }

    # ---------------------------------------------------------------------
    # Numerical validation
    # ---------------------------------------------------------------------

    def validate_numerically(
        self,
        T: sp.Matrix,
        parameter_values: Dict[sp.Symbol, float],
        x0: np.ndarray,
        *,
        t_span: Tuple[float, float] = (0.0, 10.0),
        n_eval: int = 200,
        rtol: float = 1e-8,
        atol: float = 1e-10,
        build_reduced_system: bool = True,
    ) -> Dict[str, Any]:
        """Numerically validate a candidate lumping.

        If `build_reduced_system` is True, this method attempts to:
        1) construct G(y) as a polynomial (small problems only),
        2) simulate the full and reduced systems,
        3) report max-norm discrepancy between y(t) from both simulations.

        Regardless of `build_reduced_system`, it also reports a **condition error**
        by sampling the Jacobian condition T*J*B along the trajectory.
        """
        n = self.network.n_species
        if T.cols != n:
            raise ValueError(f"T must have {n} columns")

        x_syms = self.network.x_symbols
        k_syms = self.network.rate_constants

        # Full system RHS
        F = self.network.rhs().subs(parameter_values)
        f_num = sp.lambdify(x_syms, F, modules="numpy")

        def rhs_full(t: float, x: np.ndarray) -> np.ndarray:
            val = np.array(f_num(*x), dtype=float).reshape((n,))
            return val

        t_eval = np.linspace(float(t_span[0]), float(t_span[1]), int(n_eval))
        sol_full = solve_ivp(rhs_full, t_span, np.array(x0, dtype=float), t_eval=t_eval, rtol=rtol, atol=atol)

        # Compute condition error along trajectory
        B = np.array(self.kernel_basis(T).subs(parameter_values), dtype=object)
        J = self.network.jacobian().subs(parameter_values)
        J_num = sp.lambdify(x_syms, J, modules="numpy")
        T_num = np.array(T.subs(parameter_values), dtype=float)

        cond_errs: List[float] = []
        for x in sol_full.y.T:
            Jx = np.array(J_num(*x), dtype=float)
            # Note: B may still contain symbols if T is symbolic; we guard.
            try:
                Bx = np.array(B, dtype=float)
            except Exception:
                # If cannot numeric-cast, skip condition error.
                Bx = None
            if Bx is not None and Bx.size > 0:
                M = T_num @ Jx @ Bx
                cond_errs.append(float(np.linalg.norm(M, ord=2)))

        condition_error = float(np.max(cond_errs)) if cond_errs else float("nan")

        out: Dict[str, Any] = {
            "t": sol_full.t,
            "x": sol_full.y,
            "y_from_full": (np.array(T_num @ sol_full.y, dtype=float) if sol_full.y.size else None),
            "condition_error": condition_error,
            "built_reduced_system": False,
        }

        if not build_reduced_system:
            return out

        # Attempt reduced system construction (numerical parameters substituted)
        try:
            red = self.construct_reduced_polynomial_system(T, parameter_subs=parameter_values)
            y_syms = red["y_symbols"]
            G = red["G"]

            G_num = sp.lambdify(y_syms, G, modules="numpy")

            def rhs_red(t: float, y: np.ndarray) -> np.ndarray:
                val = np.array(G_num(*y), dtype=float).reshape((T.rows,))
                return val

            y0 = np.array(T_num @ np.array(x0, dtype=float), dtype=float).reshape((T.rows,))
            sol_red = solve_ivp(rhs_red, t_span, y0, t_eval=t_eval, rtol=rtol, atol=atol)

            y_full = np.array(T_num @ sol_full.y, dtype=float)
            y_red = np.array(sol_red.y, dtype=float)

            lumping_error = float(np.max(np.linalg.norm(y_full - y_red, axis=0)))

            out.update(
                {
                    "built_reduced_system": True,
                    "y": y_red,
                    "G": G,
                    "reduced_degree": red["degree"],
                    "lumping_error": lumping_error,
                }
            )
        except Exception as exc:
            out.update({"built_reduced_system": False, "reduced_system_error": str(exc)})

        return out

    # ---------------------------------------------------------------------
    # Convenience
    # ---------------------------------------------------------------------

    @staticmethod
    def conditions_to_latex(conditions: Sequence[sp.Expr]) -> str:
        """Render conditions as a LaTeX align environment."""
        if not conditions:
            return "\\begin{align}\\end{align}"
        lines = [sp.latex(c) + " = 0" for c in conditions]
        body = " \\\\n".join(lines)
        return "\\begin{align}" + body + "\\end{align}"

    # ---------------------------------------------------------------------
    # Enumeration / reporting helpers
    # ---------------------------------------------------------------------

    def lumped_variable_expressions(
        self,
        T: sp.Matrix,
        *,
        y_prefix: str = "y",
    ) -> List[sp.Eq]:
        """Return the symbolic definitions y_i = (T x)_i as SymPy equations."""
        n = self.network.n_species
        if T.cols != n:
            raise ValueError(f"T must have {n} columns")

        x = sp.Matrix(self.network.x_symbols)
        y = sp.Matrix(T * x)
        y_syms = sp.symbols(" ".join([f"{y_prefix}{i+1}" for i in range(T.rows)]), real=True)
        return [sp.Eq(y_syms[i], sp.simplify(y[i, 0])) for i in range(T.rows)]

    def enumerate_proper_reductions(
        self,
        *,
        n_blocks: Optional[int] = None,
        max_partitions: int = 500,
        required_nonzero_rates: Optional[Sequence[Union[sp.Symbol, str]]] = None,
        solve_for_rate_constants: bool = True,
    ) -> List[Dict[str, Any]]:
        """Enumerate **proper** lumping reductions and attach reporting diagnostics.

        This is a thin wrapper around :meth:`find_proper_lumpings` that:

        - optionally filters out partitions that force selected rate constants to be
          identically zero (based on rref diagnostics), and
        - adds a stable 'kind' label for downstream reporting.
        """
        results = self.find_proper_lumpings(
            n_blocks=n_blocks,
            max_partitions=max_partitions,
            solve_for_rate_constants=solve_for_rate_constants,
        )

        if not required_nonzero_rates:
            for r in results:
                r["kind"] = "proper"
            return results

        # Normalize required_nonzero_rates into SymPy symbols.
        k_syms = self.network.rate_constants
        name_to_sym = {str(k): k for k in k_syms}
        required: List[sp.Symbol] = []
        for item in required_nonzero_rates:
            if isinstance(item, sp.Symbol):
                required.append(item)
            else:
                if str(item) not in name_to_sym:
                    raise ValueError(f"Unknown rate constant '{item}'. Known: {list(name_to_sym)}")
                required.append(name_to_sym[str(item)])

        filtered: List[Dict[str, Any]] = []
        for r in results:
            r["kind"] = "proper"
            analysis = r.get("solutions", {}).get("linear_system_analysis")
            if analysis is None:
                filtered.append(r)
                continue
            forced = set(analysis.get("forced_zero", []))
            if any(k in forced for k in required):
                continue
            filtered.append(r)

        return filtered

    def enumerate_reductions(
        self,
        *,
        include_generic: bool = True,
        include_proper: bool = True,
        proper_n_blocks: Optional[int] = None,
        max_partitions: int = 500,
        required_nonzero_rates: Optional[Sequence[Union[sp.Symbol, str]]] = None,
        include_constrained: bool = False,
        prescribed_rows: Optional[Sequence[sp.Matrix]] = None,
        n_free_rows: int = 1,
        free_param_prefix: str = "p",
        solve_for_rate_constants: bool = True,
    ) -> List[Dict[str, Any]]:
        """High-level reduction enumeration for exploratory workflows.

        In practice, one often wants to *enumerate structured candidate reductions*
        (rather than enumerate all possible linear maps T) and attach the corresponding
        critical-parameter conditions.

        - **Generic families**: Type 1 (common non-reactants) and Type 2
          (stoichiometric first integrals).
        - **Proper lumpings**: exhaustive partition search (Section 5), optionally
          filtered to avoid forcing selected rates to zero.
        - **Constrained ansatz**: build T from prescribed rows plus free rows and
          compute the linear system in rate constants.

        Notes
        -----
        - Exhaustive enumeration of *all* linear lumpings (row-echelon ansatz) is
          not included here because it can explode combinatorially and often
          requires Gröbner/primary decomposition tooling beyond SymPy.
          The provided building blocks (conditions + linear-system matrix + rref
          diagnostics) are intended to make that extension straightforward.
        """

        out: List[Dict[str, Any]] = []

        if include_generic:
            gen = self.find_generic_lumpings()

            # Type 1: projection removing common non-reactant species.
            idx = gen["common_non_reactant_indices"]
            if idx:
                keep = [i for i in range(self.network.n_species) if i not in idx]
                T1 = sp.Matrix([[1 if j == i else 0 for j in range(self.network.n_species)] for i in keep])
                out.append(
                    {
                        "kind": "generic_type1",
                        "description": "Projection eliminating common non-reactants",
                        "eliminated_indices": idx,
                        "T": T1,
                        "conditions": [],
                        "solutions": {},
                    }
                )

            # Type 2: rows are a basis of stoichiometric first integrals (constants of motion).
            integrals = gen["stoichiometric_integrals"]
            if integrals:
                T2 = sp.Matrix.vstack(*[sp.Matrix(mu) for mu in integrals])
                out.append(
                    {
                        "kind": "generic_type2",
                        "description": "Stoichiometric first integrals (constants of motion)",
                        "T": T2,
                        "conditions": [],
                        "solutions": {},
                    }
                )

        if include_proper:
            prop = self.enumerate_proper_reductions(
                n_blocks=proper_n_blocks,
                max_partitions=max_partitions,
                required_nonzero_rates=required_nonzero_rates,
                solve_for_rate_constants=solve_for_rate_constants,
            )
            out.extend(prop)

        if include_constrained:
            T, free_params = self.build_constrained_lumping_matrix(
                prescribed_rows=prescribed_rows,
                n_free_rows=n_free_rows,
                free_param_prefix=free_param_prefix,
            )
            res = self.find_critical_parameters(T, solve_for_rate_constants=solve_for_rate_constants)
            res.update(
                {
                    "kind": "constrained",
                    "description": "Constrained ansatz with prescribed rows + free parameters",
                    "free_params": free_params,
                }
            )
            out.append(res)

        return out
