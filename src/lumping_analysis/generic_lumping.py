from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union

import sympy as sp


Index = int


def _to_index_set(
    seed: Sequence[Union[int, str]],
    *,
    species_names: Optional[Sequence[str]],
) -> Set[Index]:
    out: Set[Index] = set()
    for s in seed:
        if isinstance(s, int):
            if s < 0:
                raise ValueError("Species indices must be nonnegative")
            out.add(int(s))
        elif isinstance(s, str):
            if species_names is None:
                raise ValueError("Seed contains species names but network has no species_names")
            try:
                out.add(int(species_names.index(s)))
            except ValueError as exc:
                raise ValueError(f"Unknown species name in seed: {s!r}") from exc
        else:
            raise TypeError(f"Seed items must be int or str, got {type(s)}")
    if not out:
        raise ValueError("Seed set J1 must be nonempty")
    return out


def _species_is_reactant_in_reaction(reactant: Tuple[int, ...], j: int) -> bool:
    return int(reactant[j]) != 0


def _reaction_changes_species(reactant: Tuple[int, ...], product: Tuple[int, ...], j: int) -> bool:
    # mj != rj
    return int(reactant[j]) != int(product[j])


def _all_reactant_indices(reactant: Tuple[int, ...]) -> Set[Index]:
    return {i for i, mi in enumerate(reactant) if int(mi) != 0}


@dataclass(frozen=True)
class GenericLumpingResult:
    """Result of generic lumping construction."""
    J: Tuple[int, ...]                 # final index set (sorted)
    T: sp.Matrix                       # projection matrix onto J (|J| x n)
    tagged_reactions: Tuple[int, ...]  # indices of reactions tagged as Type 1 by the algorithm
    untagged_reactions: Tuple[int, ...]
    # Optional diagnostic info:
    layers: Tuple[Tuple[int, ...], ...]


def projection_matrix_from_indices(n: int, J: Sequence[int]) -> sp.Matrix:
    """Return T that maps x -> (x_j)_{j in J} (coordinate projection)."""
    J = list(J)
    T = sp.Matrix.zeros(len(J), n)
    for row, j in enumerate(J):
        if j < 0 or j >= n:
            raise ValueError(f"Index out of range: {j} (n={n})")
        T[row, j] = sp.Integer(1)
    return T


def generic_lumping_from_seed(
    network: Any,
    seed: Sequence[Union[int, str]],
    *,
    remove_seeded_nonreactants: bool = True,
) -> GenericLumpingResult:
    """
    Construct a generic lumping map using a closure procedure.

    The steps of the algorithm are as follows:
      - start with J1 (seed indices),
      - iterate: for each j in current J_ell, scan all *untagged* reactions,
        and if reaction changes x_j (mj != rj), add ALL reactant indices of that reaction
        to the next layer set, and tag that reaction.
      - stop when the next layer set is empty.
      - output T as projection onto J = union of all layers.
      - optionally remove "seeded non-reactants" to make rank minimal.

    Parameters
    ----------
    network:
        ReactionNetwork instance. Must expose:
          - network.n_species: int
          - network.reactions: list of Reaction
        Each Reaction must expose:
          - reaction.reactant: Tuple[int,...] length n
          - reaction.product:  Tuple[int,...] length n
    seed:
        Seed species indices or names.
    remove_seeded_nonreactants:
        Implements the optional removal of "seeded non-reactants" to make rank minimal.

    Returns
    -------
    GenericLumpingResult
    """
    n = int(network.n_species)
    species_names = getattr(network, "species_names", None)

    J1 = _to_index_set(seed, species_names=species_names)

    # Reactions are referred to by integer indices.
    untagged: Set[int] = set(range(len(network.reactions)))
    tagged: List[int] = []

    layers: List[Set[int]] = [set(J1)]
    J_union: Set[int] = set(J1)

    ell = 0
    while True:
        J_ell = layers[ell]
        J_star: Set[int] = set()

        # For every j in the current layer, examine all yet untagged reactions.
        for j in sorted(J_ell):
            # We iterate over a snapshot because we may remove tags as we go.
            for ridx in list(untagged):
                r = network.reactions[ridx]
                reactant = tuple(int(v) for v in r.reactants)
                product = tuple(int(v) for v in r.products)

                # If mj == rj, continue (species j not involved or is a catalyst).
                if not _reaction_changes_species(reactant, product, j):
                    continue

                # Else: tag this reaction as Type 1 and add ALL reactant indices.
                tagged.append(ridx)
                untagged.remove(ridx)

                reactant_idxs = _all_reactant_indices(reactant)
                for i in reactant_idxs:
                    if i not in J_union:
                        J_star.add(i)

        if not J_star:
            break

        layers.append(J_star)
        J_union |= J_star
        ell += 1

    # Optional final “remove seeded non-reactants” rank-minimization step:
    # If a seeded species never appears as a reactant in ANY reaction, it can be dropped.
    if remove_seeded_nonreactants:
        # Determine all reactant species across the whole network.
        all_reactants: Set[int] = set()
        for r in network.reactions:
            all_reactants |= _all_reactant_indices(tuple(int(v) for v in r.reactants))

        # Remove only those that were part of the original seed.
        drop = {j for j in J1 if j not in all_reactants}
        if drop:
            J_union = set(j for j in J_union if j not in drop)

    J_sorted = tuple(sorted(J_union))
    T = projection_matrix_from_indices(n, J_sorted)

    return GenericLumpingResult(
        J=J_sorted,
        T=T,
        tagged_reactions=tuple(sorted(set(tagged))),
        untagged_reactions=tuple(sorted(untagged)),
        layers=tuple(tuple(sorted(L)) for L in layers),
    )