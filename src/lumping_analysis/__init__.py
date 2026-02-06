"""Top-level package API for lumping_analysis.

This package implements symbolic tools for analyzing **linear lumping** of
parameter-dependent mass action reaction networks, as described in the paper
"Lumping of reaction networks: Generic and critical parameters" (draft dated
Feb 4, 2026).

Public API:
- Reaction, ReactionNetwork
- LumpingAnalyzer
- Built-in example networks
"""

from .reaction import Reaction
from .network import ReactionNetwork
from .parser import ReactionParser
from .analyzer import LumpingAnalyzer
from .singular import SingularIdeal
from .report import (
    ReductionReportOptions,
    format_reduction_report,
    format_reduction_result,
)
from .examples import (
    michaelis_menten_network,
    three_species_linear_network,
    gpl_replication_network,
)

__all__ = [
    "Reaction",
    "ReactionNetwork",
    "ReactionParser",
    "LumpingAnalyzer",
    "SingularIdeal",
    "ReductionReportOptions",
    "format_reduction_report",
    "format_reduction_result",
    "michaelis_menten_network",
    "three_species_linear_network",
    "gpl_replication_network",
]
