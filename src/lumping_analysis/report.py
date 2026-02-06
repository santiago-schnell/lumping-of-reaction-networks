"""Human-readable reporting utilities.

This module provides lightweight (dependency-free) helpers to produce readable
console / Markdown reports from the Python API:

- candidate lumping maps (T matrices),
- critical-parameter conditions, and
- induced linear relations among rate constants when the condition system is
  linear in the rate constants.

Nothing here is required for the core algebra; it is strictly presentation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

import sympy as sp

from .network import ReactionNetwork


def _expr_to_str(e: sp.Expr) -> str:
    """Stable string for SymPy expressions in reports."""
    try:
        return sp.sstr(e)
    except Exception:
        return str(e)


def format_lumping_map(
    network: ReactionNetwork,
    T: sp.Matrix,
    *,
    y_prefix: str = "y",
) -> List[str]:
    """Format y = T x as a list of strings "y1 = ..."."""
    x = sp.Matrix(network.x_symbols)
    y = sp.simplify(T * x)

    lines: List[str] = []
    for i in range(T.rows):
        lhs = f"{y_prefix}{i+1}"
        rhs = _expr_to_str(y[i, 0])
        lines.append(f"{lhs} = {rhs}")
    return lines


def format_conditions(conditions: Sequence[sp.Expr], *, max_items: int = 10) -> List[str]:
    """Format polynomial equalities c=0."""
    out: List[str] = []
    for c in list(conditions)[: int(max_items)]:
        out.append(f"{_expr_to_str(sp.factor(c))} = 0")
    if len(conditions) > max_items:
        out.append(f"... ({len(conditions) - max_items} more)")
    return out


def format_linear_relations(
    analysis: Optional[Dict[str, Any]],
    *,
    max_items: int = 10,
) -> List[str]:
    """Format the rref-derived linear relations among rate constants."""
    if not analysis:
        return []
    rels = analysis.get("relations", [])
    forced = analysis.get("forced_zero", [])

    out: List[str] = []
    for r in list(rels)[: int(max_items)]:
        out.append(f"{_expr_to_str(r)} = 0")

    if forced:
        forced_str = ", ".join(str(v) for v in forced)
        out.append(f"forced zero: {forced_str}")

    if len(rels) > max_items:
        out.append(f"... ({len(rels) - max_items} more relations)")

    return out


@dataclass
class ReductionReportOptions:
    """Tunable knobs for report verbosity."""

    max_conditions: int = 12
    max_relations: int = 12
    include_T_matrix: bool = False
    y_prefix: str = "y"


def format_reduction_result(
    network: ReactionNetwork,
    result: Dict[str, Any],
    *,
    options: Optional[ReductionReportOptions] = None,
) -> str:
    """Format a single reduction result (from LumpingAnalyzer.* methods)."""
    opt = options or ReductionReportOptions()

    kind = result.get("kind", "unknown")
    desc = result.get("description", "")
    lines: List[str] = []

    lines.append(f"### {kind}")
    if desc:
        lines.append(desc)

    # Proper lumpings have blocks.
    if kind == "proper":
        blocks = result.get("blocks")
        if blocks:
            lines.append(f"Blocks: {blocks}")

    T = result.get("T")
    if isinstance(T, sp.MatrixBase):
        lines.append("Lumping map:")
        lines.extend(["  " + s for s in format_lumping_map(network, sp.Matrix(T), y_prefix=opt.y_prefix)])
        if opt.include_T_matrix:
            lines.append("T =")
            lines.append(sp.pretty(sp.Matrix(T)))

    conditions = result.get("conditions", []) or []
    if conditions:
        lines.append("Conditions:")
        lines.extend(["  " + s for s in format_conditions(conditions, max_items=opt.max_conditions)])

    analysis = result.get("solutions", {}).get("linear_system_analysis")
    rel_lines = format_linear_relations(analysis, max_items=opt.max_relations)
    if rel_lines:
        lines.append("Linear relations (rref diagnostics):")
        lines.extend(["  " + s for s in rel_lines])

    # Show free parameter symbols for constrained ansatz.
    if kind == "constrained":
        fp = result.get("free_params", [])
        if fp:
            lines.append("Free parameters: " + ", ".join(str(s) for s in fp))

    return "\n".join(lines)


def format_reduction_report(
    network: ReactionNetwork,
    results: Iterable[Dict[str, Any]],
    *,
    options: Optional[ReductionReportOptions] = None,
) -> str:
    """Format a multi-result report."""
    opt = options or ReductionReportOptions()
    blocks: List[str] = []
    for res in results:
        blocks.append(format_reduction_result(network, res, options=opt))
        blocks.append("")
    return "\n".join(blocks).rstrip() + "\n"
