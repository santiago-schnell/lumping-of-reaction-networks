"""Enumerate and report candidate reductions for reversible Michaelis--Menten.

This script illustrates an **exploratory workflow**:

- enumerate structured candidate reductions (generic families + proper lumpings), and
- print a compact report (lumping maps, critical conditions, and induced linear
  relations among rates when applicable).

For n=4 species, exhaustive partition search for proper lumpings is inexpensive.

Run:
    python examples/enumerate_reductions_michaelis_menten.py
"""

from __future__ import annotations

from lumping_analysis import (
    LumpingAnalyzer,
    ReductionReportOptions,
    format_reduction_report,
    michaelis_menten_network,
)


def main() -> None:
    net = michaelis_menten_network()
    an = LumpingAnalyzer(net)

    # Enumerate proper lumpings into 3 blocks (dimension reduction 4 -> 3),
    # while filtering out candidates that necessarily force any rate constant
    # to be identically zero (based on rref diagnostics).
    results = an.enumerate_reductions(
        include_generic=True,
        include_proper=True,
        proper_n_blocks=3,
        max_partitions=10_000,
        required_nonzero_rates=net.rate_constants,
    )

    report = format_reduction_report(
        net,
        results,
        options=ReductionReportOptions(max_conditions=12, max_relations=12, include_T_matrix=False),
    )
    print(report)


if __name__ == "__main__":
    main()
