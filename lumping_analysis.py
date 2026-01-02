"""
lumping_analysis.py

Computer Algebra Tools for Linear Lumping of Mass Action Reaction Networks

This module accompanies the paper:
    "Lumping of reaction networks: Generic settings vs. critical parameters"
    J.E., V.R., S.S., S.W.
    SIAM Journal on Applied Dynamical Systems (2025)

The module provides tools to:
1. Define reaction networks symbolically with mass action kinetics
2. Find generic lumpings (Type 1 and Type 2) that work for all parameters
3. Identify critical parameters where non-trivial lumpings become available
4. Compute reduced systems at critical parameter values
5. Validate lumpings numerically
6. Export equations to LaTeX

Requirements:
    Python >= 3.8
    sympy >= 1.12
    numpy >= 1.20
    scipy >= 1.7

Basic usage:
    >>> from lumping_analysis import michaelis_menten_network, LumpingAnalyzer
    >>> network = michaelis_menten_network()
    >>> analyzer = LumpingAnalyzer(network)
    >>> generic = analyzer.find_generic_lumpings()

Author: Santiago Schnell (santiago.schnell@dartmouth.edu)
License: MIT
Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "Santiago Schnell"
__email__ = "santiago.schnell@dartmouth.edu"

import sympy as sp
from sympy import (symbols, Matrix, zeros, eye, simplify, expand, factor,
                   solve, Poly, latex, Symbol, Integer, Rational)
import numpy as np
from scipy.integrate import odeint
from typing import List, Tuple, Dict, Optional, Set, Union, Callable
from dataclasses import dataclass, field
from itertools import combinations
import re
import warnings


# ============================================================
# CORE CLASSES
# ============================================================

@dataclass
class Reaction:
    """
    Represents a single mass action reaction.
    
    A reaction has the form:
        m₁X₁ + m₂X₂ + ... + mₙXₙ  →[k]  r₁X₁ + r₂X₂ + ... + rₙXₙ
    
    where mᵢ are reactant coefficients, rᵢ are product coefficients,
    and k is the rate constant.
    
    Attributes:
        reactant_coeffs: Tuple of stoichiometric coefficients for reactants (mᵢ)
        product_coeffs: Tuple of stoichiometric coefficients for products (rᵢ)
        rate_symbol: Symbolic rate constant k
        name: Optional human-readable name for the reaction
        
    Example:
        >>> k1 = sp.Symbol('k_1', positive=True)
        >>> rxn = Reaction((1, 1, 0, 0), (0, 0, 1, 0), k1, "S + E -> C")
    """
    reactant_coeffs: Tuple[int, ...]
    product_coeffs: Tuple[int, ...]
    rate_symbol: sp.Symbol
    name: Optional[str] = None
    
    @property
    def n_species(self) -> int:
        """Number of species in the reaction."""
        return len(self.reactant_coeffs)
    
    @property
    def reaction_vector(self) -> Matrix:
        """
        Returns the stoichiometric vector v = r - m.
        
        This vector represents the net change in each species
        when the reaction occurs once.
        """
        return Matrix([r - m for m, r in 
                      zip(self.reactant_coeffs, self.product_coeffs)])
    
    @property
    def monomial(self) -> sp.Expr:
        """
        Returns the mass action monomial x^m = x₁^m₁ · x₂^m₂ · ... · xₙ^mₙ.
        """
        x = symbols(f'x1:{self.n_species + 1}')
        result = sp.Integer(1)
        for i, m in enumerate(self.reactant_coeffs):
            if m > 0:
                result *= x[i]**m
        return result
    
    def rhs_contribution(self, x_symbols: List[sp.Symbol]) -> Matrix:
        """
        Returns the contribution k · x^m · v to the RHS of the ODE system.
        
        Args:
            x_symbols: List of symbolic concentration variables
            
        Returns:
            Column vector representing this reaction's contribution to dx/dt
        """
        monomial = sp.Integer(1)
        for i, m in enumerate(self.reactant_coeffs):
            if m > 0:
                monomial *= x_symbols[i]**m
        return self.rate_symbol * monomial * self.reaction_vector
    
    def to_string(self, species_names: List[str] = None) -> str:
        """
        Convert reaction to human-readable string.
        
        Args:
            species_names: Optional list of species names
            
        Returns:
            String like "A + B -> C"
        """
        if species_names is None:
            species_names = [f'X{i+1}' for i in range(self.n_species)]
        
        def format_side(coeffs):
            terms = []
            for i, c in enumerate(coeffs):
                if c == 0:
                    continue
                elif c == 1:
                    terms.append(species_names[i])
                else:
                    terms.append(f'{c}{species_names[i]}')
            return ' + '.join(terms) if terms else '0'
        
        lhs = format_side(self.reactant_coeffs)
        rhs = format_side(self.product_coeffs)
        return f"{lhs} -> {rhs}"
    
    def to_latex(self, species_names: List[str] = None) -> str:
        """
        Convert reaction to LaTeX string.
        
        Args:
            species_names: Optional list of species names (can include LaTeX)
            
        Returns:
            LaTeX string like "A + B \\xrightarrow{k_1} C"
        """
        if species_names is None:
            species_names = [f'X_{{{i+1}}}' for i in range(self.n_species)]
        
        def format_side(coeffs):
            terms = []
            for i, c in enumerate(coeffs):
                if c == 0:
                    continue
                elif c == 1:
                    terms.append(species_names[i])
                else:
                    terms.append(f'{c}{species_names[i]}')
            return ' + '.join(terms) if terms else '\\emptyset'
        
        lhs = format_side(self.reactant_coeffs)
        rhs = format_side(self.product_coeffs)
        rate = latex(self.rate_symbol)
        return f"{lhs} \\xrightarrow{{{rate}}} {rhs}"


@dataclass  
class ReactionNetwork:
    """
    Represents a mass action reaction network.
    
    A reaction network consists of n species and a set of reactions,
    giving rise to the ODE system:
        dx/dt = F(x, k) = Σᵢ kᵢ · x^mᵢ · vᵢ
    
    Attributes:
        n_species: Number of chemical species
        reactions: List of Reaction objects
        species_names: Optional list of species names for display
        
    Example:
        >>> network = ReactionNetwork(4, reactions, ['S', 'E', 'C', 'P'])
        >>> F = network.rhs()  # Get symbolic RHS
        >>> J = network.jacobian()  # Get Jacobian matrix
    """
    n_species: int
    reactions: List[Reaction]
    species_names: Optional[List[str]] = None
    _x: Tuple[sp.Symbol, ...] = field(default=None, repr=False)
    _rate_symbols: List[sp.Symbol] = field(default=None, repr=False)
    
    def __post_init__(self):
        if self.species_names is None:
            self.species_names = [f'X_{i+1}' for i in range(self.n_species)]
        
        # Create symbolic concentration variables
        self._x = symbols(f'x1:{self.n_species + 1}')
        self._rate_symbols = [r.rate_symbol for r in self.reactions]
    
    @property
    def x(self) -> Tuple[sp.Symbol, ...]:
        """Symbolic concentration variables (x₁, x₂, ..., xₙ)."""
        return self._x
    
    @property
    def rate_symbols(self) -> List[sp.Symbol]:
        """List of rate constant symbols."""
        return self._rate_symbols
    
    @classmethod
    def from_string(cls, reaction_string: str, 
                    species_names: List[str] = None) -> 'ReactionNetwork':
        """
        Create a ReactionNetwork from a string representation.
        
        Supported formats:
            "A + B -> C"           Irreversible reaction
            "A + B <-> C"          Reversible reaction (auto-generates k and k₋)
            "A + B ->[k1] C"       With explicit rate constant
            "2A -> B"              Stoichiometric coefficients
            
        Multiple reactions separated by newlines or semicolons.
        
        Args:
            reaction_string: String containing reaction definitions
            species_names: Optional list to fix species ordering
            
        Returns:
            ReactionNetwork object
            
        Example:
            >>> network = ReactionNetwork.from_string('''
            ...     S + E <-> C
            ...     C -> E + P
            ... ''')
        """
        return ReactionParser.parse(reaction_string, species_names)
    
    def rhs(self) -> Matrix:
        """
        Returns the full RHS F(x,k) as a symbolic column vector.
        
        The system dx/dt = F(x,k) where F is the sum of contributions
        from all reactions.
        
        Returns:
            n×1 Matrix containing symbolic expressions for dx/dt
        """
        result = zeros(self.n_species, 1)
        for reaction in self.reactions:
            result += reaction.rhs_contribution(self._x)
        return result
    
    def jacobian(self) -> Matrix:
        """
        Returns the Jacobian DF(x,k) symbolically.
        
        The (i,j) entry is ∂Fᵢ/∂xⱼ.
        
        Returns:
            n×n Matrix containing symbolic partial derivatives
        """
        F = self.rhs()
        return F.jacobian(self._x)
    
    def stoichiometric_matrix(self) -> Matrix:
        """
        Returns the stoichiometric matrix S.
        
        Columns are reaction vectors vᵢ = rᵢ - mᵢ.
        The stoichiometric subspace is the column space of S.
        
        Returns:
            n×r Matrix where r is the number of reactions
        """
        return Matrix([[r.reaction_vector[i] for r in self.reactions] 
                       for i in range(self.n_species)])
    
    def stoichiometric_first_integrals(self) -> List[Matrix]:
        """
        Returns a basis for stoichiometric first integrals.
        
        These are vectors μ such that μᵀ·v = 0 for all reaction vectors v.
        They correspond to conservation laws: μᵀ·x = constant along solutions.
        
        Returns:
            List of column vectors forming a basis for ker(Sᵀ)
        """
        S = self.stoichiometric_matrix()
        null_space = S.T.nullspace()
        return null_space
    
    def rhs_numeric(self, k_values: Dict[sp.Symbol, float]) -> Callable:
        """
        Return a numerical function for the RHS.
        
        Args:
            k_values: Dictionary mapping rate symbols to numerical values
            
        Returns:
            Function f(x, t) suitable for scipy.integrate.odeint
        """
        F = self.rhs()
        F_substituted = F.subs(k_values)
        
        # Convert to list of expressions for lambdify
        F_list = [F_substituted[i] for i in range(self.n_species)]
        f_lambdified = sp.lambdify(self._x, F_list, modules=['numpy'])
        
        def rhs_func(x, t):
            result = f_lambdified(*x)
            return np.array(result, dtype=float).flatten()
        
        return rhs_func
    
    def to_latex(self, align: bool = True) -> str:
        """
        Export the full ODE system as LaTeX.
        
        Args:
            align: If True, use align environment; otherwise plain equations
            
        Returns:
            LaTeX string for the ODE system
        """
        F = self.rhs()
        lines = []
        
        # Create substitution dict: x1->species_name, etc.
        x_to_name = {self._x[i]: sp.Symbol(self.species_names[i]) 
                     for i in range(self.n_species)}
        
        for i, name in enumerate(self.species_names):
            rhs_expr = F[i].subs(x_to_name)
            rhs_tex = latex(rhs_expr)
            
            if align:
                lines.append(f"\\dot{{{name}}} &= {rhs_tex}")
            else:
                lines.append(f"\\dot{{{name}}} = {rhs_tex}")
        
        if align:
            return "\\begin{align}\n" + " \\\\\n".join(lines) + "\n\\end{align}"
        else:
            return "\n".join(lines)
    
    def reactions_to_latex(self) -> str:
        """
        Export all reactions as LaTeX.
        
        Returns:
            LaTeX string with all reactions
        """
        lines = []
        for rxn in self.reactions:
            lines.append(rxn.to_latex(self.species_names))
        return " \\\\\n".join(lines)
    
    def summary(self) -> str:
        """
        Return a text summary of the network.
        
        Returns:
            Multi-line string describing the network
        """
        lines = [
            f"Reaction Network Summary",
            f"=" * 40,
            f"Species ({self.n_species}): {', '.join(self.species_names)}",
            f"Reactions ({len(self.reactions)}):"
        ]
        for i, rxn in enumerate(self.reactions):
            lines.append(f"  {i+1}. {rxn.to_string(self.species_names)}")
        
        integrals = self.stoichiometric_first_integrals()
        lines.append(f"\nStoichiometric first integrals ({len(integrals)}):")
        for i, mu in enumerate(integrals):
            terms = []
            for j, coeff in enumerate(mu):
                if coeff != 0:
                    if coeff == 1:
                        terms.append(self.species_names[j])
                    elif coeff == -1:
                        terms.append(f"-{self.species_names[j]}")
                    else:
                        terms.append(f"{coeff}·{self.species_names[j]}")
            integral_str = ' + '.join(terms).replace('+ -', '- ')
            lines.append(f"  μ_{i+1} = {integral_str}")
        
        return "\n".join(lines)


# ============================================================
# REACTION STRING PARSER
# ============================================================

class ReactionParser:
    """
    Parse reaction strings into ReactionNetwork objects.
    
    Supports:
        - Species names (letters, optionally followed by numbers/subscripts)
        - Stoichiometric coefficients (2A, 3B)
        - Reversible reactions (<-> or <=>)
        - Irreversible reactions (-> or =>)
        - Explicit rate constants (->[k1])
        - Multiple reactions (newline or semicolon separated)
    """
    
    SPECIES_PATTERN = r'([A-Za-z][A-Za-z0-9_]*)'
    COEFF_SPECIES_PATTERN = r'(\d*)([A-Za-z][A-Za-z0-9_]*)'
    ARROW_PATTERN = r'(<->|<=>|->|=>)'
    RATE_PATTERN = r'\[([^\]]+)\]'
    
    @classmethod
    def parse(cls, reaction_string: str, 
              species_names: List[str] = None) -> ReactionNetwork:
        """Parse a reaction string into a ReactionNetwork."""
        
        # Split into individual reactions
        reaction_string = reaction_string.strip()
        reaction_lines = re.split(r'[;\n]', reaction_string)
        reaction_lines = [line.strip() for line in reaction_lines if line.strip()]
        
        # First pass: collect all species
        all_species = set()
        for line in reaction_lines:
            species = cls._extract_species(line)
            all_species.update(species)
        
        # Determine species ordering
        if species_names is not None:
            for sp in all_species:
                if sp not in species_names:
                    raise ValueError(f"Species '{sp}' not in provided species_names")
            ordered_species = species_names
        else:
            ordered_species = sorted(all_species)
        
        n_species = len(ordered_species)
        species_index = {name: i for i, name in enumerate(ordered_species)}
        
        # Second pass: parse reactions
        reactions = []
        rate_counter = [1]
        
        for line in reaction_lines:
            parsed = cls._parse_single_reaction(line, species_index, n_species, rate_counter)
            reactions.extend(parsed)
        
        return ReactionNetwork(n_species, reactions, ordered_species)
    
    @classmethod
    def _extract_species(cls, line: str) -> Set[str]:
        """Extract all species names from a reaction line."""
        line = re.sub(cls.RATE_PATTERN, '', line)
        line = re.sub(cls.ARROW_PATTERN, ' ', line)
        matches = re.findall(cls.COEFF_SPECIES_PATTERN, line)
        return {match[1] for match in matches}
    
    @classmethod
    def _parse_single_reaction(cls, line: str, species_index: Dict[str, int],
                                n_species: int, rate_counter: List[int]) -> List[Reaction]:
        """Parse a single reaction line."""
        reactions = []
        
        if '<->' in line or '<=>' in line:
            is_reversible = True
            arrow = '<->' if '<->' in line else '<=>'
        else:
            is_reversible = False
            arrow = '->' if '->' in line else '=>'
        
        parts = re.split(cls.ARROW_PATTERN, line)
        if len(parts) < 3:
            raise ValueError(f"Invalid reaction format: {line}")
        
        lhs = parts[0].strip()
        rhs = parts[2].strip()
        
        rate_match = re.search(cls.RATE_PATTERN, line)
        if rate_match:
            rate_name = rate_match.group(1)
            k_fwd = symbols(rate_name, positive=True)
            rhs = re.sub(cls.RATE_PATTERN, '', rhs).strip()
        else:
            k_fwd = symbols(f'k_{rate_counter[0]}', positive=True)
            rate_counter[0] += 1
        
        reactant_coeffs = cls._parse_side(lhs, species_index, n_species)
        product_coeffs = cls._parse_side(rhs, species_index, n_species)
        
        reactions.append(Reaction(
            tuple(reactant_coeffs),
            tuple(product_coeffs),
            k_fwd,
            name=line.strip()
        ))
        
        if is_reversible:
            k_rev = symbols(f'k_{{-{rate_counter[0]-1}}}', positive=True)
            reactions.append(Reaction(
                tuple(product_coeffs),
                tuple(reactant_coeffs),
                k_rev,
                name=f"reverse of {line.strip()}"
            ))
        
        return reactions
    
    @classmethod
    def _parse_side(cls, side: str, species_index: Dict[str, int], 
                    n_species: int) -> List[int]:
        """Parse one side of a reaction (reactants or products)."""
        coeffs = [0] * n_species
        
        side = side.strip()
        if side == '0' or side == '' or side.lower() == 'null':
            return coeffs
        
        terms = [t.strip() for t in side.split('+')]
        
        for term in terms:
            if not term:
                continue
            
            match = re.match(cls.COEFF_SPECIES_PATTERN, term)
            if match:
                coeff_str, species = match.groups()
                coeff = int(coeff_str) if coeff_str else 1
                
                if species in species_index:
                    coeffs[species_index[species]] = coeff
                else:
                    raise ValueError(f"Unknown species: {species}")
        
        return coeffs


# ============================================================
# LUMPING ANALYZER
# ============================================================

class LumpingAnalyzer:
    """
    Analyzes linear lumping for mass action networks.
    
    Implements the theory from the paper:
        - Proposition 3.1: Type 1/Type 2 classification for single reactions
        - Proposition 3.4: Generic lumping characterization
        - Lemma 4.1: T·DF(x,k)·B = 0 condition for critical parameters
        - Proposition 5.1: Column sum criterion for proper lumping
    
    Attributes:
        network: The ReactionNetwork to analyze
        n: Number of species
        x: Symbolic concentration variables
        
    Example:
        >>> network = michaelis_menten_network()
        >>> analyzer = LumpingAnalyzer(network)
        >>> generic = analyzer.find_generic_lumpings()
        >>> result = analyzer.find_critical_parameters(T)
    """
    
    def __init__(self, network: ReactionNetwork):
        """
        Initialize the analyzer.
        
        Args:
            network: ReactionNetwork to analyze
        """
        self.network = network
        self.n = network.n_species
        self.x = network.x
        
    def check_single_reaction_type(self, reaction: Reaction) -> Dict:
        """
        Classify invariant subspaces for a single reaction (Proposition 3.1).
        
        For a single reaction, invariant subspaces are either:
            - Type 1: Spanned by non-reactant species (species with mᵢ = 0)
            - Type 2: Containing the reaction vector v
        
        Args:
            reaction: A Reaction object to analyze
            
        Returns:
            Dictionary with keys:
                'type1_indices': List of non-reactant species indices
                'type1_basis': Basis vectors for Type 1 subspace
                'type2_basis': Basis for Type 2 subspace (contains v)
                'reaction_vector': The reaction vector v
        """
        n = reaction.n_species
        m = reaction.reactant_coeffs
        v = reaction.reaction_vector
        
        # Type 1: Non-reactant species (m_i = 0)
        non_reactant_indices = [i for i, mi in enumerate(m) if mi == 0]
        type1_basis = [Matrix([1 if j == i else 0 for j in range(n)]) 
                       for i in non_reactant_indices]
        
        # Type 2: Contains reaction vector v
        type2_basis = [v] if not v.equals(zeros(n, 1)) else []
        
        return {
            'type1_indices': non_reactant_indices,
            'type1_basis': type1_basis,
            'type2_basis': type2_basis,
            'reaction_vector': v
        }
    
    def find_generic_lumpings(self) -> Dict:
        """
        Find all generic lumping maps for the network (Section 3).
        
        Generic lumpings work for all parameter values and come in two types:
            - Type 1: Based on common non-reactant species
            - Type 2: Based on stoichiometric first integrals
        
        Returns:
            Dictionary with keys:
                'common_non_reactants': Set of species indices that are 
                    never reactants in any reaction
                'stoichiometric_integrals': List of first integral vectors
                'type1_possible': Whether Type 1 lumping exists
                'type2_possible': Whether Type 2 lumping exists
        """
        results = {
            'common_non_reactants': set(range(self.n)),
            'stoichiometric_integrals': [],
            'type1_possible': True,
            'type2_possible': True
        }
        
        # Find common non-reactant species (Type 1)
        for reaction in self.network.reactions:
            reactant_indices = {i for i, m in enumerate(reaction.reactant_coeffs) if m > 0}
            non_reactants = set(range(self.n)) - reactant_indices
            results['common_non_reactants'] &= non_reactants
        
        if not results['common_non_reactants']:
            results['type1_possible'] = False
        
        # Find stoichiometric first integrals (Type 2)
        results['stoichiometric_integrals'] = self.network.stoichiometric_first_integrals()
        
        if not results['stoichiometric_integrals']:
            results['type2_possible'] = False
        
        return results
    
    def build_constrained_lumping_matrix(self, 
                                          prescribed_rows: List[Matrix] = None,
                                          n_free_rows: int = 1,
                                          free_param_prefix: str = 'p') -> Tuple[Matrix, List[sp.Symbol]]:
        """
        Build a lumping matrix with prescribed rows and free parameters.
        
        Useful for constrained lumping (Remark 4.3) where some rows of T
        are prescribed (e.g., stoichiometric first integrals).
        
        Args:
            prescribed_rows: List of row vectors to include as fixed rows
            n_free_rows: Number of additional rows with free parameters
            free_param_prefix: Prefix for free parameter symbols
            
        Returns:
            Tuple of (T matrix, list of free parameter symbols)
            
        Example:
            >>> integrals = network.stoichiometric_first_integrals()
            >>> T, params = analyzer.build_constrained_lumping_matrix(
            ...     prescribed_rows=integrals, n_free_rows=1)
        """
        if prescribed_rows is None:
            prescribed_rows = []
        
        # Create free parameters
        free_params = []
        free_rows = []
        
        for i in range(n_free_rows):
            row = []
            for j in range(self.n):
                param = symbols(f'{free_param_prefix}_{i+1}{j+1}')
                free_params.append(param)
                row.append(param)
            free_rows.append(Matrix([row]))
        
        # Combine prescribed and free rows
        all_rows = [Matrix(r).T if r.shape[0] > r.shape[1] else r 
                    for r in prescribed_rows]
        all_rows.extend(free_rows)
        
        T = Matrix([list(row) for row in all_rows])
        
        return T, free_params
    
    def find_critical_parameters(self, T: Matrix, 
                                  free_params: List[sp.Symbol] = None,
                                  simplify_conditions: bool = True) -> Dict:
        """
        Find critical parameters for a given lumping map T (Lemma 4.1).
        
        A parameter value k* is critical for T if T defines a valid
        lumping map for the specialization at k*. This requires:
            T · DF(x, k*) · B = 0  for all x
        where B is a matrix whose columns span ker(T).
        
        Args:
            T: Lumping matrix (e × n) where e < n
            free_params: Optional list of free parameters in T
            simplify_conditions: Whether to factor/simplify conditions
        
        Returns:
            Dictionary with keys:
                'kernel_basis': Matrix B with columns spanning ker(T)
                'product_TDFxB': The matrix T·DF(x,k)·B
                'conditions': List of polynomial conditions on parameters
                'solutions': Dict with solution information
                'rate_symbols': List of rate constant symbols
                'free_params': List of free parameters in T
        """
        e = T.shape[0]  # reduced dimension
        n = T.shape[1]  # original dimension
        
        assert n == self.n, f"T has {n} columns but network has {self.n} species"
        
        # Compute kernel basis B
        B = self._compute_kernel_basis(T)
        
        if B is None or B.shape[1] == 0:
            return {'error': 'T has trivial kernel (full rank n)'}
        
        # Compute Jacobian
        DF = self.network.jacobian()
        
        # Compute T·DF·B
        product = T * DF * B
        
        # Extract polynomial conditions
        conditions = self._extract_polynomial_conditions(product)
        
        if simplify_conditions:
            conditions = [factor(c) for c in conditions]
            conditions = list(set(conditions))
            conditions = [c for c in conditions if c != 0]
        
        # Solve for critical parameters
        rate_symbols = self.network.rate_symbols
        all_params = list(rate_symbols)
        if free_params:
            all_params.extend(free_params)
        
        solutions = self._solve_conditions(conditions, all_params)
        
        return {
            'kernel_basis': B,
            'product_TDFxB': product,
            'conditions': conditions,
            'solutions': solutions,
            'rate_symbols': rate_symbols,
            'free_params': free_params or []
        }
    
    def _compute_kernel_basis(self, T: Matrix) -> Optional[Matrix]:
        """Compute a basis for ker(T)."""
        null_space = T.nullspace()
        if not null_space:
            return None
        return Matrix([list(v) for v in null_space]).T
    
    def _extract_polynomial_conditions(self, expr_matrix: Matrix) -> List[sp.Expr]:
        """
        Extract polynomial conditions from T·DF·B = 0.
        
        Since this must hold for all x, we extract coefficients of all
        monomials in x and require each to be zero.
        """
        conditions = []
        
        for i in range(expr_matrix.shape[0]):
            for j in range(expr_matrix.shape[1]):
                expr = expand(expr_matrix[i, j])
                
                if expr == 0:
                    continue
                
                try:
                    poly = sp.Poly(expr, self.x)
                    for coeff in poly.coeffs():
                        if coeff != 0:
                            conditions.append(coeff)
                except Exception:
                    conditions.append(expr)
        
        unique_conditions = list(set(simplify(c) for c in conditions))
        return [c for c in unique_conditions if c != 0]
    
    def _solve_conditions(self, conditions: List[sp.Expr], 
                          params: List[sp.Symbol]) -> Dict:
        """Solve the polynomial conditions for critical parameters."""
        if not conditions:
            return {'status': 'all_parameters_critical', 'solutions': []}
        
        try:
            solutions = solve(conditions, params, dict=True)
            return {
                'status': 'solved',
                'solutions': solutions,
                'conditions': conditions
            }
        except Exception as e:
            return {
                'status': 'conditions_only',
                'conditions': conditions,
                'error': str(e)
            }
    
    def conditions_to_latex(self, conditions: List[sp.Expr]) -> str:
        """
        Export critical parameter conditions as LaTeX.
        
        Args:
            conditions: List of polynomial conditions
            
        Returns:
            LaTeX align environment with conditions
        """
        lines = []
        for cond in conditions:
            lines.append(f"{latex(cond)} &= 0")
        return "\\begin{align}\n" + " \\\\\n".join(lines) + "\n\\end{align}"
    
    def validate_numerically(self, T: Matrix, 
                              k_values: Dict[sp.Symbol, float],
                              x0: np.ndarray,
                              t_span: Tuple[float, float] = (0, 10),
                              n_points: int = 100,
                              tolerance: float = 1e-6) -> Dict:
        """
        Numerically validate that T defines a valid lumping.
        
        Integrates the ODE system and checks whether T·F(x,k) depends
        only on y = Tx, as required for exact lumping.
        
        Args:
            T: Lumping matrix
            k_values: Numerical values for rate constants
            x0: Initial condition (n-dimensional array)
            t_span: Time span for integration (t_start, t_end)
            n_points: Number of time points for integration
            tolerance: Error tolerance for validation
            
        Returns:
            Dictionary with keys:
                'is_valid': Boolean indicating if lumping is valid
                'lumping_error': Maximum error in lumping condition
                'condition_error': Error in T·DF·B = 0 condition
                'max_variance': Variance measure for consistency
                'x_trajectory': Array of x values over time
                'y_trajectory': Array of y = Tx values over time
                't': Time points
        """
        t = np.linspace(t_span[0], t_span[1], n_points)
        T_np = np.array(T.tolist(), dtype=float)
        
        # Get numerical RHS and integrate
        rhs_func = self.network.rhs_numeric(k_values)
        x_traj = odeint(rhs_func, x0, t)
        
        # Compute y trajectory
        y_traj = x_traj @ T_np.T
        
        # Compute lumping error: ||T·dX/dt - d(TX)/dt||
        dxdt = np.gradient(x_traj, t, axis=0)
        dydt = np.gradient(y_traj, t, axis=0)
        T_dxdt = dxdt @ T_np.T
        lumping_error = np.abs(T_dxdt - dydt).max()
        
        # Check condition T·DF·B = 0 numerically
        B = self._compute_kernel_basis(T)
        if B is not None:
            B_np = np.array(B.tolist(), dtype=float)
            DF = self.network.jacobian()
            DF_sub = DF.subs(k_values)
            
            condition_errors = []
            for x in x_traj[::10]:
                x_dict = {self.x[i]: x[i] for i in range(self.n)}
                DF_val = np.array(DF_sub.subs(x_dict).tolist(), dtype=float)
                TDFxB = T_np @ DF_val @ B_np
                condition_errors.append(np.abs(TDFxB).max())
            
            max_condition_error = max(condition_errors)
        else:
            max_condition_error = 0
        
        # Check consistency: same y should give same T·F
        from collections import defaultdict
        
        F = self.network.rhs()
        F_sub = F.subs(k_values)
        F_list = [F_sub[i] for i in range(self.n)]
        F_lambdified = sp.lambdify(self.x, F_list, modules=['numpy'])
        
        TF_traj = []
        for x in x_traj:
            Fx = np.array(F_lambdified(*x), dtype=float).flatten()
            TFx = T_np @ Fx
            TF_traj.append(TFx)
        TF_traj = np.array(TF_traj)
        
        y_rounded = np.round(y_traj, decimals=6)
        y_to_TF = defaultdict(list)
        for i, y in enumerate(y_rounded):
            key = tuple(y)
            y_to_TF[key].append(TF_traj[i])
        
        max_variance = 0
        for key, tf_values in y_to_TF.items():
            if len(tf_values) > 1:
                variance = np.var(tf_values, axis=0).max()
                max_variance = max(max_variance, variance)
        
        # Include condition_error in validity check
        is_valid = (max_condition_error < tolerance and 
                    max_variance < tolerance and 
                    lumping_error < tolerance)
        
        return {
            'is_valid': is_valid,
            'lumping_error': lumping_error,
            'condition_error': max_condition_error,
            'max_variance': max_variance,
            'x_trajectory': x_traj,
            'y_trajectory': y_traj,
            't': t,
            'tolerance': tolerance
        }
    
    def compute_reduced_system(self, T: Matrix, 
                                critical_params: Dict[sp.Symbol, sp.Expr] = None) -> Dict:
        """
        Compute information about the reduced system at critical parameters.
        
        Args:
            T: Lumping matrix
            critical_params: Dict mapping rate symbols to critical values
        
        Returns:
            Dictionary with:
                'TF': The product T·F(x,k)
                'y_symbols': Symbols for reduced variables
                'y_definitions': Expressions for y in terms of x
        """
        F = self.network.rhs()
        
        if critical_params:
            F = F.subs(critical_params)
        
        TF = T * F
        y = symbols(f'y1:{T.shape[0] + 1}')
        
        return {
            'TF': TF,
            'y_symbols': y,
            'y_definitions': [sum(T[i,j] * self.x[j] for j in range(self.n)) 
                             for i in range(T.shape[0])],
            'note': 'Express TF in terms of y by substitution/elimination'
        }


# ============================================================
# NETWORK FACTORY FUNCTIONS
# ============================================================

def create_reversible_reaction(reactants: Tuple[int, ...], 
                                products: Tuple[int, ...],
                                k_fwd: sp.Symbol, 
                                k_rev: sp.Symbol) -> List[Reaction]:
    """
    Create forward and reverse reactions for a reversible reaction.
    
    Args:
        reactants: Tuple of reactant stoichiometric coefficients
        products: Tuple of product stoichiometric coefficients
        k_fwd: Forward rate constant symbol
        k_rev: Reverse rate constant symbol
        
    Returns:
        List containing forward and reverse Reaction objects
    """
    return [
        Reaction(reactants, products, k_fwd),
        Reaction(products, reactants, k_rev)
    ]


def michaelis_menten_network() -> ReactionNetwork:
    """
    Create the reversible Michaelis-Menten network.
    
    Reaction scheme:
        S + E ⇌ C ⇌ E + P
    
    Species:
        S (substrate), E (enzyme), C (complex), P (product)
    
    Stoichiometric first integrals:
        μ₁ = S - E + P  (substrate-product balance)
        μ₂ = E + C      (total enzyme conservation)
        
    This network is analyzed in Example 4.3 of the paper.
    
    Returns:
        ReactionNetwork object
    """
    k1, km1, k2, km2 = symbols('k_1 k_{-1} k_2 k_{-2}', positive=True)
    
    reactions = [
        Reaction((1, 1, 0, 0), (0, 0, 1, 0), k1, "S + E -> C"),
        Reaction((0, 0, 1, 0), (1, 1, 0, 0), km1, "C -> S + E"),
        Reaction((0, 0, 1, 0), (0, 1, 0, 1), k2, "C -> E + P"),
        Reaction((0, 1, 0, 1), (0, 0, 1, 0), km2, "E + P -> C"),
    ]
    
    return ReactionNetwork(4, reactions, ['S', 'E', 'C', 'P'])


def three_species_linear_network() -> ReactionNetwork:
    """
    Create a three-species linear (monomolecular) network.
    
    Reaction scheme:
        X₁ ⇌ X₂ ⇌ X₃
    
    Stoichiometric first integral:
        μ = X₁ + X₂ + X₃  (total mass conservation)
        
    This network is analyzed in Example 4.2 of the paper.
    The critical parameter condition is: k₁·k₋₂ = k₋₁·k₂
    
    Returns:
        ReactionNetwork object
    """
    k1, km1, k2, km2 = symbols('k_1 k_{-1} k_2 k_{-2}', positive=True)
    
    reactions = [
        Reaction((1, 0, 0), (0, 1, 0), k1, "X1 -> X2"),
        Reaction((0, 1, 0), (1, 0, 0), km1, "X2 -> X1"),
        Reaction((0, 1, 0), (0, 0, 1), k2, "X2 -> X3"),
        Reaction((0, 0, 1), (0, 1, 0), km2, "X3 -> X2"),
    ]
    
    return ReactionNetwork(3, reactions, ['X_1', 'X_2', 'X_3'])


def gpl_replication_network() -> ReactionNetwork:
    """
    Create the GPL self-replication network from Section 6.
    
    From: Gijima and Peacock-López, "A dynamic study of biochemical 
    self-replication", Mathematics 8, 1042 (2020).
    
    Species:
        A (X₁), B (X₂), P (X₃), Iₐ (X₄), Iᵦ (X₅), I (X₆)
    
    Reaction scheme:
        A + P ⇌ Iₐ       (k₁, k₋₁)
        B + Iₐ ⇌ I       (k₂, k₋₂)
        B + P ⇌ Iᵦ       (k₃, k₋₃)
        A + Iᵦ ⇌ I       (k₄, k₋₄)
        I ⇌ 2P           (k₅, k₋₅)
    
    Critical parameters: k₂ = k₄ = 0
    
    At critical parameters, the hidden conservation law emerges:
        y₁ = P + Iₐ + Iᵦ + r·I
    where r = (2k₅ + k₋₂ + k₋₄)/(k₅ + k₋₂ + k₋₄)
    
    Returns:
        ReactionNetwork object
    """
    k1, km1 = symbols('k_1 k_{-1}', positive=True)
    k2, km2 = symbols('k_2 k_{-2}', positive=True)
    k3, km3 = symbols('k_3 k_{-3}', positive=True)
    k4, km4 = symbols('k_4 k_{-4}', positive=True)
    k5, km5 = symbols('k_5 k_{-5}', positive=True)
    
    reactions = [
        Reaction((1, 0, 1, 0, 0, 0), (0, 0, 0, 1, 0, 0), k1, "A + P -> Ia"),
        Reaction((0, 0, 0, 1, 0, 0), (1, 0, 1, 0, 0, 0), km1, "Ia -> A + P"),
        Reaction((0, 1, 0, 1, 0, 0), (0, 0, 0, 0, 0, 1), k2, "B + Ia -> I"),
        Reaction((0, 0, 0, 0, 0, 1), (0, 1, 0, 1, 0, 0), km2, "I -> B + Ia"),
        Reaction((0, 1, 1, 0, 0, 0), (0, 0, 0, 0, 1, 0), k3, "B + P -> Ib"),
        Reaction((0, 0, 0, 0, 1, 0), (0, 1, 1, 0, 0, 0), km3, "Ib -> B + P"),
        Reaction((1, 0, 0, 0, 1, 0), (0, 0, 0, 0, 0, 1), k4, "A + Ib -> I"),
        Reaction((0, 0, 0, 0, 0, 1), (1, 0, 0, 0, 1, 0), km4, "I -> A + Ib"),
        Reaction((0, 0, 0, 0, 0, 1), (0, 0, 2, 0, 0, 0), k5, "I -> 2P"),
        Reaction((0, 0, 2, 0, 0, 0), (0, 0, 0, 0, 0, 1), km5, "2P -> I"),
    ]
    
    return ReactionNetwork(6, reactions, ['A', 'B', 'P', 'I_a', 'I_b', 'I'])


def self_complementarity_network() -> ReactionNetwork:
    """
    Create the self-complementarity replication network.
    
    This network models template-directed self-replication where
    complementary binding leads to template formation.
    
    Species:
        A (aminoadenosine), E (imide ester), AE (complex), T (template),
        AT, AET, T₂ (template dimer), ET
    
    Reaction scheme:
        A + E ⇌ AE → T
        A + T ⇌ AT
        E + AT ⇌ AET → T₂
        E + T ⇌ ET
        A + ET ⇌ AET
        T₂ ⇌ 2T
        
    Stoichiometric first integrals:
        φ₁ = A + AE + T + 2AT + 2AET + 2T₂ + ET
        φ₂ = E + AE + T + AT + 2AET + 2T₂ + 2ET
    
    Returns:
        ReactionNetwork object
    """
    k1, km1, k2 = symbols('k_1 k_{-1} k_2', positive=True)
    k3, km3 = symbols('k_3 k_{-3}', positive=True)
    k4, km4, k5 = symbols('k_4 k_{-4} k_5', positive=True)
    k6, km6 = symbols('k_6 k_{-6}', positive=True)
    k7, km7 = symbols('k_7 k_{-7}', positive=True)
    k8, km8 = symbols('k_8 k_{-8}', positive=True)
    
    # Species: A=0, E=1, AE=2, T=3, AT=4, AET=5, T2=6, ET=7
    reactions = [
        Reaction((1,1,0,0,0,0,0,0), (0,0,1,0,0,0,0,0), k1, "A + E -> AE"),
        Reaction((0,0,1,0,0,0,0,0), (1,1,0,0,0,0,0,0), km1, "AE -> A + E"),
        Reaction((0,0,1,0,0,0,0,0), (0,0,0,1,0,0,0,0), k2, "AE -> T"),
        Reaction((1,0,0,1,0,0,0,0), (0,0,0,0,1,0,0,0), k3, "A + T -> AT"),
        Reaction((0,0,0,0,1,0,0,0), (1,0,0,1,0,0,0,0), km3, "AT -> A + T"),
        Reaction((0,1,0,0,1,0,0,0), (0,0,0,0,0,1,0,0), k4, "E + AT -> AET"),
        Reaction((0,0,0,0,0,1,0,0), (0,1,0,0,1,0,0,0), km4, "AET -> E + AT"),
        Reaction((0,0,0,0,0,1,0,0), (0,0,0,0,0,0,1,0), k5, "AET -> T2"),
        Reaction((0,1,0,1,0,0,0,0), (0,0,0,0,0,0,0,1), k6, "E + T -> ET"),
        Reaction((0,0,0,0,0,0,0,1), (0,1,0,1,0,0,0,0), km6, "ET -> E + T"),
        Reaction((1,0,0,0,0,0,0,1), (0,0,0,0,0,1,0,0), k7, "A + ET -> AET"),
        Reaction((0,0,0,0,0,1,0,0), (1,0,0,0,0,0,0,1), km7, "AET -> A + ET"),
        Reaction((0,0,0,0,0,0,1,0), (0,0,0,2,0,0,0,0), k8, "T2 -> 2T"),
        Reaction((0,0,0,2,0,0,0,0), (0,0,0,0,0,0,1,0), km8, "2T -> T2"),
    ]
    
    return ReactionNetwork(8, reactions, 
                          ['A', 'E', 'AE', 'T', 'AT', 'AET', 'T_2', 'ET'])


def substrate_inhibition_network() -> ReactionNetwork:
    """
    Create a substrate inhibition network.
    
    Reaction scheme:
        S + E ⇌ C → E + P
        S + C ⇌ C₂  (dead-end inhibited complex)
    
    At high substrate concentrations, substrate binds to the ES complex
    forming an inactive ternary complex, leading to inhibition.
    
    Returns:
        ReactionNetwork object
    """
    k1, km1, k2 = symbols('k_1 k_{-1} k_2', positive=True)
    k3, km3 = symbols('k_3 k_{-3}', positive=True)
    
    reactions = [
        Reaction((1, 1, 0, 0, 0), (0, 0, 1, 0, 0), k1, "S + E -> C"),
        Reaction((0, 0, 1, 0, 0), (1, 1, 0, 0, 0), km1, "C -> S + E"),
        Reaction((0, 0, 1, 0, 0), (0, 1, 0, 1, 0), k2, "C -> E + P"),
        Reaction((1, 0, 1, 0, 0), (0, 0, 0, 0, 1), k3, "S + C -> C2"),
        Reaction((0, 0, 0, 0, 1), (1, 0, 1, 0, 0), km3, "C2 -> S + C"),
    ]
    
    return ReactionNetwork(5, reactions, ['S', 'E', 'C', 'P', 'C_2'])


def competitive_inhibition_network() -> ReactionNetwork:
    """
    Create a competitive inhibition network.
    
    Reaction scheme:
        S + E ⇌ C → E + P
        E + I ⇌ EI
    
    The inhibitor I competes with substrate S for binding to the enzyme.
    
    Returns:
        ReactionNetwork object
    """
    k1, km1, k2 = symbols('k_1 k_{-1} k_2', positive=True)
    k3, km3 = symbols('k_3 k_{-3}', positive=True)
    
    reactions = [
        Reaction((1, 1, 0, 0, 0, 0), (0, 0, 1, 0, 0, 0), k1, "S + E -> C"),
        Reaction((0, 0, 1, 0, 0, 0), (1, 1, 0, 0, 0, 0), km1, "C -> S + E"),
        Reaction((0, 0, 1, 0, 0, 0), (0, 1, 0, 1, 0, 0), k2, "C -> E + P"),
        Reaction((0, 1, 0, 0, 1, 0), (0, 0, 0, 0, 0, 1), k3, "E + I -> EI"),
        Reaction((0, 0, 0, 0, 0, 1), (0, 1, 0, 0, 1, 0), km3, "EI -> E + I"),
    ]
    
    return ReactionNetwork(6, reactions, ['S', 'E', 'C', 'P', 'I', 'EI'])


def double_phosphorylation_network() -> ReactionNetwork:
    """
    Create a double phosphorylation network (MAPK cascade motif).
    
    Reaction scheme:
        S + E ⇌ ES → Sₚ + E
        Sₚ + E ⇌ ESₚ → Sₚₚ + E
    
    This represents distributive (non-processive) phosphorylation
    where the kinase must rebind for each phosphorylation step.
    
    Returns:
        ReactionNetwork object
    """
    k1, km1, k2 = symbols('k_1 k_{-1} k_2', positive=True)
    k3, km3, k4 = symbols('k_3 k_{-3} k_4', positive=True)
    
    reactions = [
        Reaction((1, 0, 0, 1, 0, 0), (0, 0, 0, 0, 1, 0), k1, "S + E -> ES"),
        Reaction((0, 0, 0, 0, 1, 0), (1, 0, 0, 1, 0, 0), km1, "ES -> S + E"),
        Reaction((0, 0, 0, 0, 1, 0), (0, 1, 0, 1, 0, 0), k2, "ES -> Sp + E"),
        Reaction((0, 1, 0, 1, 0, 0), (0, 0, 0, 0, 0, 1), k3, "Sp + E -> ESp"),
        Reaction((0, 0, 0, 0, 0, 1), (0, 1, 0, 1, 0, 0), km3, "ESp -> Sp + E"),
        Reaction((0, 0, 0, 0, 0, 1), (0, 0, 1, 1, 0, 0), k4, "ESp -> Spp + E"),
    ]
    
    return ReactionNetwork(6, reactions, ['S', 'S_p', 'S_{pp}', 'E', 'ES', 'ES_p'])


# ============================================================
# MODULE INFO
# ============================================================

def list_available_networks() -> Dict[str, str]:
    """
    List all available network factory functions.
    
    Returns:
        Dictionary mapping function names to descriptions
    """
    return {
        'michaelis_menten_network': 'Reversible Michaelis-Menten (Example 4.3)',
        'three_species_linear_network': 'Three-species linear chain (Example 4.2)',
        'gpl_replication_network': 'GPL self-replication (Section 6)',
        'self_complementarity_network': 'Self-complementarity replication',
        'substrate_inhibition_network': 'Substrate inhibition',
        'competitive_inhibition_network': 'Competitive inhibition',
        'double_phosphorylation_network': 'Double phosphorylation (MAPK motif)',
    }


if __name__ == '__main__':
    print(f"Lumping Analysis Module v{__version__}")
    print("=" * 50)
    print("\nAvailable network factory functions:")
    for name, desc in list_available_networks().items():
        print(f"  {name}(): {desc}")
    print("\nFor usage examples, see README.md or the examples/ folder.")
