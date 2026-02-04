"""Enumerate proper lumping partitions for reversible Michaelis--Menten.

For n=4 species there are only 15 set partitions, so exhaustive search is cheap.

Run:
    python examples/search_proper_lumpings_michaelis_menten.py
"""

from __future__ import annotations

import sympy as sp

from lumping_analysis import LumpingAnalyzer, michaelis_menten_network


def main() -> None:
    net = michaelis_menten_network()
    an = LumpingAnalyzer(net)

    # Enumerate all partitions into 3 blocks.
    results = an.find_proper_lumpings(n_blocks=3, max_partitions=10_000, solve_for_rate_constants=True)

    print(f"Found {len(results)} partitions into 3 blocks (including trivial reorderings):")
    for res in results:
        blocks = res["blocks"]
        conds = res["conditions"]
        if not conds:
            continue
        print("\nBlocks:", blocks)
        for c in conds:
            print("  ", sp.factor(c), "= 0")


if __name__ == "__main__":
    main()
