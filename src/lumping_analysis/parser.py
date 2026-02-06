from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import sympy as sp

from .reaction import Reaction


# A single term like "2A" or "2 A" or "A".
_TERM_RE = re.compile(r"^\s*(?:(\d+)\s*)?([A-Za-z0-9_]+)\s*$")

# Supported arrow tokens. We normalize these to either "->" or "<->".
_ARROW_RE = re.compile(r"(<=>|<->|=>|->)")


def _parse_complex(complex_str: str) -> Dict[str, int]:
    """Parse a complex string like '2A + B' into {'A':2, 'B':1}.

    Accepted:
    - '0' or '' for the empty complex
    - terms separated by '+'
    - coefficients as nonnegative integers (e.g. '2A', '2 A')

    Returns
    -------
    dict
        Mapping species names to nonnegative integer coefficients.
    """
    s = complex_str.strip()
    if s == "" or s == "0":
        return {}

    parts = [p.strip() for p in s.split("+") if p.strip()]
    coeffs: Dict[str, int] = {}
    for part in parts:
        m = _TERM_RE.match(part)
        if not m:
            raise ValueError(f"Could not parse complex term: '{part}'")
        c_str, name = m.group(1), m.group(2)
        c = int(c_str) if c_str is not None else 1
        if c < 0:
            raise ValueError("Stoichiometric coefficients must be nonnegative")
        coeffs[name] = coeffs.get(name, 0) + c
    return coeffs


def _complex_to_vector(complex_dict: Dict[str, int], species_order: Sequence[str]) -> Tuple[int, ...]:
    return tuple(int(complex_dict.get(name, 0)) for name in species_order)


def _consume_leading_rate_brackets(s: str) -> Tuple[List[str], str]:
    """Consume leading [ ... ] blocks and return (tokens, remainder).

    Supported forms:
        "[k1] C"
        "[k1][km1] C"
        "[k1, km1] C"
        "[k1; km1] C"

    Returns
    -------
    tokens:
        A flat list of tokens (strings) found in brackets.
    remainder:
        The remainder of the string after removing leading brackets.
    """
    tokens: List[str] = []
    rest = s.strip()

    while rest.startswith("["):
        end = rest.find("]")
        if end == -1:
            raise ValueError(f"Unclosed '[' in rate specification: '{s}'")
        inside = rest[1:end].strip()
        if inside:
            # Allow comma or semicolon separated tokens inside a single bracket.
            parts = [p.strip() for p in re.split(r"[;,]", inside) if p.strip()]
            tokens.extend(parts)
        rest = rest[end + 1 :].strip()

    return tokens, rest


@dataclass
class ReactionParser:
    """Parse reaction strings into mass-action `Reaction` objects.

    Supported arrows
    ---------------
    - irreversible: '->' or '=>'
    - reversible: '<->' or '<=>'

    Rate constants (optional)
    -------------------------
    You may specify rate constants in square brackets immediately after the arrow.
    Examples:
    - 'A + B ->[k1] C'
    - 'A + B <->[k1][km1] C'
    - 'A + B <->[k1, km1] C'

    If no rates are provided, the parser auto-generates them as:
    - irreversible:  k1, k2, ...
    - reversible:    k1/km1, k2/km2, ...  (one index per reversible pair)
    """

    rate_prefix: str = "k"
    assume_positive_rates: bool = True

    def parse_network(self, text: str, species_names: Optional[Sequence[str]] = None):
        # Split lines (semicolon or newline)
        raw_lines: List[str] = []
        for chunk in text.split(";"):
            raw_lines.extend(chunk.splitlines())
        lines = [ln.strip() for ln in raw_lines if ln.strip() and not ln.strip().startswith("#")]
        if not lines:
            raise ValueError("No reactions found in input")

        # First pass: collect species if not provided.
        if species_names is None:
            species_set = set()
            for ln in lines:
                lhs, _arrow, rhs, _rates = self._split_reaction_line(ln)
                species_set |= set(_parse_complex(lhs).keys())
                species_set |= set(_parse_complex(rhs).keys())
            species_order = sorted(species_set)
        else:
            species_order = list(species_names)

        reactions: List[Reaction] = []
        pair_idx = 1
        for ln in lines:
            lhs_str, arrow, rhs_str, rate_tokens = self._split_reaction_line(ln)
            lhs = _parse_complex(lhs_str)
            rhs = _parse_complex(rhs_str)

            if arrow == "<->":
                # Rate token handling:
                #  - []:      auto k{i}, km{i}
                #  - [kf]:    forward fixed, reverse auto km{i}
                #  - [kf][kr] or [kf,kr]: both fixed
                if len(rate_tokens) == 0:
                    kf = self._make_rate_symbol(f"{self.rate_prefix}{pair_idx}")
                    kr = self._make_rate_symbol(f"{self.rate_prefix}m{pair_idx}")
                elif len(rate_tokens) == 1:
                    kf = self._make_rate_symbol(rate_tokens[0])
                    kr = self._make_rate_symbol(f"{self.rate_prefix}m{pair_idx}")
                elif len(rate_tokens) == 2:
                    kf = self._make_rate_symbol(rate_tokens[0])
                    kr = self._make_rate_symbol(rate_tokens[1])
                else:
                    raise ValueError(
                        f"Too many rate tokens for reversible reaction '{ln}'. "
                        "Use at most two (forward, reverse)."
                    )

                reactions.append(
                    Reaction(
                        reactants=_complex_to_vector(lhs, species_order),
                        products=_complex_to_vector(rhs, species_order),
                        rate=kf,
                    )
                )
                reactions.append(
                    Reaction(
                        reactants=_complex_to_vector(rhs, species_order),
                        products=_complex_to_vector(lhs, species_order),
                        rate=kr,
                    )
                )
                pair_idx += 1

            elif arrow == "->":
                if len(rate_tokens) == 0:
                    kf = self._make_rate_symbol(f"{self.rate_prefix}{pair_idx}")
                elif len(rate_tokens) == 1:
                    kf = self._make_rate_symbol(rate_tokens[0])
                else:
                    raise ValueError(
                        f"Too many rate tokens for irreversible reaction '{ln}'. "
                        "Use at most one."
                    )

                reactions.append(
                    Reaction(
                        reactants=_complex_to_vector(lhs, species_order),
                        products=_complex_to_vector(rhs, species_order),
                        rate=kf,
                    )
                )
                pair_idx += 1
            else:
                raise ValueError(f"Unsupported arrow '{arrow}' in line '{ln}'")

        from .network import ReactionNetwork  # local import to avoid circular import

        return ReactionNetwork(n_species=len(species_order), reactions=reactions, species_names=list(species_order))

    def _make_rate_symbol(self, name: str) -> sp.Symbol:
        """Create a SymPy symbol for a rate constant."""
        # Keep the symbol name exactly as provided; this is important when
        # users pass LaTeX-like names such as k_{-1}.
        if self.assume_positive_rates:
            return sp.Symbol(name, positive=True)
        return sp.Symbol(name)

    @staticmethod
    def _split_reaction_line(line: str) -> Tuple[str, str, str, List[str]]:
        """Split a reaction line into (lhs, arrow, rhs, rate_tokens)."""
        ln = line.strip()
        m = _ARROW_RE.search(ln)
        if not m:
            raise ValueError(f"No supported arrow found in line: '{line}'")

        arrow_raw = m.group(1)
        arrow = "<->" if arrow_raw in {"<->", "<=>"} else "->"

        lhs = ln[: m.start()].strip()
        rest = ln[m.end() :].strip()

        rate_tokens, rhs = _consume_leading_rate_brackets(rest)
        rhs = rhs.strip()
        if rhs == "":
            raise ValueError(f"Missing RHS complex in line: '{line}'")

        return lhs, arrow, rhs, rate_tokens
