from __future__ import annotations

"""Interoperability helpers for the computer algebra system **Singular**.

This package keeps all core computations in Python/SymPy, but for larger
problems (especially *ideal* operations like Groebner bases and primary
decomposition) it can be advantageous to export polynomial generators to
Singular.

This module provides:

- a small `SingularIdeal` data structure,
- conversion of SymPy expressions (polynomials) into Singular syntax, and
- optional execution of Singular via subprocess (if installed).

Nothing in this module requires Singular at *import time*; only the
`SingularIdeal.run()` method assumes a `Singular` executable is available.

Notes
-----
- Singular variable names must be valid identifiers. If your SymPy symbols
  contain characters like braces or minus signs (e.g. `k_{-1}`), we sanitize
  names deterministically during export.
- The exporters assume the expressions are polynomials/rational functions.
  If expressions contain non-polynomial constructs (e.g. `sin`, `exp`), export
  will fail.
"""

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union
import re
import subprocess

import sympy as sp


def _sanitize_var_name(name: str) -> str:
    """Convert an arbitrary string into a safe Singular identifier."""
    # Keep only alphanumeric + underscore.
    s = re.sub(r"[^0-9A-Za-z_]", "_", name)
    s = re.sub(r"_+", "_", s).strip("_")
    if not s:
        s = "v"
    if s[0].isdigit():
        s = f"v_{s}"
    return s


def make_symbol_name_map(symbols: Sequence[sp.Symbol]) -> Dict[sp.Symbol, str]:
    """Create a deterministic, collision-free mapping sympy Symbol -> Singular name."""
    used: Dict[str, int] = {}
    out: Dict[sp.Symbol, str] = {}

    # Deterministic order by string representation.
    for sym in sorted(symbols, key=lambda s: str(s)):
        base = _sanitize_var_name(str(sym))
        if base not in used:
            used[base] = 0
            out[sym] = base
        else:
            used[base] += 1
            out[sym] = f"{base}_{used[base]}"

    return out


def sympy_to_singular(expr: sp.Expr, name_map: Dict[sp.Symbol, str]) -> str:
    """Convert a SymPy expression to Singular syntax.

    Parameters
    ----------
    expr:
        SymPy expression (ideally a polynomial).
    name_map:
        Mapping of SymPy symbols to sanitized Singular variable names.

    Returns
    -------
    str
        A Singular-readable string.
    """
    # Replace symbols with safe names by xreplace.
    repl = {s: sp.Symbol(name_map[s]) for s in name_map}
    expr2 = sp.expand(expr).xreplace(repl)

    s = sp.sstr(expr2)

    # Singular uses '^' for exponentiation.
    s = s.replace("**", "^")
    # Remove spaces to keep scripts compact.
    s = s.replace(" ", "")
    return s


@dataclass(frozen=True)
class SingularIdeal:
    """A polynomial ideal intended for export to Singular."""

    generators: Tuple[sp.Expr, ...]
    variables: Tuple[sp.Symbol, ...]
    characteristic: int = 0
    monomial_order: str = "dp"  # 'dp' = degree reverse lexicographic (global order)

    @classmethod
    def from_generators(
        cls,
        generators: Sequence[sp.Expr],
        *,
        variables: Optional[Sequence[sp.Symbol]] = None,
        characteristic: int = 0,
        monomial_order: str = "dp",
    ) -> "SingularIdeal":
        if variables is None:
            syms: set = set()
            for g in generators:
                syms |= set(sp.expand(g).free_symbols)
            variables = sorted(syms, key=lambda s: str(s))
        return cls(tuple(generators), tuple(variables), int(characteristic), str(monomial_order))

    def name_map(self) -> Dict[sp.Symbol, str]:
        return make_symbol_name_map(list(self.variables))

    def to_singular_script(
        self,
        *,
        ring_name: str = "R",
        ideal_name: str = "I",
        compute_groebner: bool = False,
        primary_decomposition: bool = False,
        eliminate: Optional[Sequence[sp.Symbol]] = None,
        comment: Optional[str] = None,
    ) -> str:
        """Render a Singular script defining the ring and ideal.

        Parameters
        ----------
        compute_groebner:
            If True, append commands computing and printing a Groebner basis.
        primary_decomposition:
            If True, append commands loading `primdec.lib` and calling `primdecGTZ`.
            (This can be expensive; primarily intended as a starting point.)
        eliminate:
            If provided, append an elimination command eliminating those variables.
            Singular's `eliminate` expects the *product* of variables to eliminate.
        comment:
            Optional comment header (will be prefixed with `// ` on each line).
        """
        nm = self.name_map()
        vars_sing = [nm[s] for s in self.variables]
        gens_sing = [sympy_to_singular(g, nm) for g in self.generators]

        lines: List[str] = []
        if comment:
            for ln in str(comment).splitlines():
                lines.append(f"// {ln}")
        lines.append(f"ring {ring_name} = {self.characteristic},({','.join(vars_sing)}),{self.monomial_order};")
        if not gens_sing:
            lines.append(f"ideal {ideal_name} = 0;")
        else:
            lines.append(f"ideal {ideal_name} = {','.join(gens_sing)};")
        lines.append("")  # spacer

        if eliminate:
            elim_names = [nm.get(v, _sanitize_var_name(str(v))) for v in eliminate]
            # eliminate(I, x*y*z) eliminates x,y,z
            prod = "*".join(elim_names) if elim_names else "1"
            lines.append(f"ideal {ideal_name}_elim = eliminate({ideal_name}, {prod});")
            lines.append(f"print({ideal_name}_elim);")
            lines.append("")

        if compute_groebner:
            lines.append(f"ideal {ideal_name}_gb = groebner({ideal_name});")
            lines.append(f"print({ideal_name}_gb);")
            lines.append("")

        if primary_decomposition:
            lines.append('LIB "primdec.lib";')
            lines.append(f"list {ideal_name}_pd = primdecGTZ({ideal_name});")
            lines.append(f"print({ideal_name}_pd);")
            lines.append("")

        return "\n".join(lines)

    def run(
        self,
        *,
        singular_executable: str = "Singular",
        script: Optional[str] = None,
        timeout: int = 60,
    ) -> str:
        """Run Singular on the given script and return stdout.

        This is a convenience wrapper around `subprocess.run`. It is *optional*:
        many users will prefer to copy-paste the generated script into their own
        Singular environment.

        Parameters
        ----------
        singular_executable:
            Name or path of the Singular binary.
        script:
            If provided, run this script instead of `to_singular_script()`.
        timeout:
            Timeout in seconds.

        Returns
        -------
        stdout as a string.
        """
        if script is None:
            script = self.to_singular_script()

        proc = subprocess.run(
            [singular_executable, "-q"],
            input=script.encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=int(timeout),
            check=False,
        )
        out = proc.stdout.decode("utf-8", errors="replace")
        err = proc.stderr.decode("utf-8", errors="replace")
        if proc.returncode != 0:
            raise RuntimeError(
                f"Singular exited with code {proc.returncode}.\nSTDERR:\n{err}\nSTDOUT:\n{out}"
            )
        # Some Singular warnings are printed on stderr even on success; append them.
        if err.strip():
            out = out + "\n\n// STDERR\n" + err
        return out
