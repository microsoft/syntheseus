"""Code related to starting molecules match metric (called exact set-wise match in FusionRetro)."""

from __future__ import annotations

import itertools
import typing
from collections.abc import Iterable
from typing import Optional

from syntheseus.search.chem import Molecule
from syntheseus.search.graph.and_or import ANDOR_NODE, AndNode, AndOrGraph, OrNode

T = typing.TypeVar("T")


def is_route_with_starting_mols(
    graph: AndOrGraph,
    starting_mols: set[Molecule],
    forbidden_nodes: typing.Optional[set[ANDOR_NODE]] = None,
) -> bool:
    """Checks whether there is a route in graph matching the starting mols."""
    forbidden_nodes = forbidden_nodes or set()
    return _is_route_with_starting_mols(
        graph,
        graph.root_node,
        starting_mols,
        forbidden_nodes,
    )


def split_into_subsets(A: list[T], k: int) -> Iterable[list[list[T]]]:
    """
    Enumerate all possible ways to create k subsets from a list A such that none of the k subsets are empty,
    the union of the sets is A, and the order of the subsets *does* matter.

    This function is easiest to explain by example:

    A single partition is just the list A itself.

    >>> list(split_into_subsets([1, 2], 1))
    >>> [[[1, 2]]]

    For k=2, there are 4 possible subsets meaning 16 pairs of subsets,
    but only 8 of them have non-empty subsets which include every element once.

    >>> list(split_into_subsets([1, 2,], 2))
    >>> [[[1], [2]], [[1], [1,2]], [[2], [1]], [[2], [1,2]], [[1,2], [1]], [[1,2], [2]]]
    >>> [[[1], [2]], [[1], [1, 2]], [[2], [1]], [[2], [1, 2]], [[1, 2], [1]], [[1, 2], [2]], [[1, 2], [1, 2]]]

    The implementation just uses itertools.combinations and itertools.products to enumerate all possible partions,
    and simply rejects those which do not sum up to the entire set.
    It is not very efficient for large A or large k, so use with caution.

    NOTE: the efficiency of this method could definitely be improved later.
    """

    # Check 1: elements of list are unique
    assert len(set(A)) == len(A)

    # Check 2: k is valid
    assert k >= 1

    # Base case: list is empty
    if len(A) == 0:
        return

    # Iterate through all subsets
    power_set_non_empty = itertools.chain.from_iterable(
        itertools.combinations(A, r) for r in range(1, len(A) + 1)
    )
    for subsets in itertools.product(power_set_non_empty, repeat=k):
        if set(itertools.chain.from_iterable(subsets)) == set(A):
            yield [list(s) for s in subsets]


def _is_solvable_from_starting_mols(
    graph: AndOrGraph,
    starting_mols: set[Molecule],
    forbidden_nodes: Optional[set[ANDOR_NODE]] = None,
) -> dict[ANDOR_NODE, bool]:
    """Get whether each node is solvable only from a specified set of starting molecules."""
    forbidden_nodes = forbidden_nodes or set()

    # Which nodes are solvable because they contain a starting molecule?
    node_to_contains_start_mol = {
        n: (isinstance(n, OrNode) and n.mol in starting_mols) for n in graph.nodes()
    }
    node_to_solvable = {n: False for n in graph.nodes()}

    # Do passes through all nodes
    update_happened = True
    while update_happened:
        update_happened = False
        for n in graph.nodes():
            successors_are_solvable = [node_to_solvable[c] for c in graph.successors(n)]
            if n in forbidden_nodes:
                new_solvable = False  # regardless of successors, forbidden nodes are not solvable
            elif isinstance(n, OrNode):
                new_solvable = any(successors_are_solvable) or node_to_contains_start_mol[n]
            elif isinstance(n, AndNode):
                new_solvable = all(successors_are_solvable)
            else:
                raise ValueError

            if new_solvable != node_to_solvable[n]:
                node_to_solvable[n] = new_solvable
                update_happened = True

    return node_to_solvable


def _starting_mols_under_each_node(graph: AndOrGraph) -> dict[ANDOR_NODE, set[Molecule]]:
    """Get set of molecules reachable under each node in the graph."""

    # Initialize to empty sets, except for nodes with purchasable mols
    node_to_mols: dict[ANDOR_NODE, set[Molecule]] = {n: set() for n in graph.nodes()}
    for n in graph.nodes():
        if isinstance(n, OrNode):
            node_to_mols[n].add(n.mol)

    # Do passes through all nodes
    update_happened = True
    while update_happened:
        update_happened = False
        for n in graph.nodes():
            for c in graph.successors(n):
                if not (node_to_mols[c] <= node_to_mols[n]):
                    node_to_mols[n].update(node_to_mols[c])
                    update_happened = True

    return node_to_mols


def _is_route_with_starting_mols(
    graph: AndOrGraph,
    start_node: OrNode,
    starting_mols: set[Molecule],
    forbidden_nodes: set[ANDOR_NODE],
    node_to_solvable: Optional[dict[ANDOR_NODE, bool]] = None,
    node_to_reachable_starting_mols: Optional[dict[ANDOR_NODE, set[Molecule]]] = None,
) -> bool:
    """
    Recursive method to check whether there is a route in the graph,
    starting from `start_node` and excluding `forbidden_nodes`,
    whose leaves and exactly `starting_mols`.

    To prune the search early, we use the `node_to_solvable` dictionary,
    which contains True if a node *might* be solvable from only the starting molecules.
    """
    assert start_node in graph

    # Compute node to solvable if not provided
    if node_to_solvable is None:
        node_to_solvable = _is_solvable_from_starting_mols(graph, starting_mols, forbidden_nodes)
    if node_to_reachable_starting_mols is None:
        node_to_reachable_starting_mols = _starting_mols_under_each_node(graph)

    # Base case 1: starting mols is empty
    if len(starting_mols) == 0:
        return False

    # Base case 2: start node is forbidden
    if start_node in forbidden_nodes:
        return False

    # Base case 3: start node not solvable
    if (
        not node_to_solvable[start_node]
        or not node_to_reachable_starting_mols[start_node] >= starting_mols
    ):
        return False

    # Base case 4: there is just one starting molecule and this OrNode contains it.
    if len(starting_mols) == 1 and list(starting_mols)[0] == start_node.mol:
        return True

    # Main case: the required starting molecules are reachable,
    # but we just need to check whether they are reachable within a single synthesis route.
    # We do this by explicitly trying to find this synthesis route.
    for rxn_child in graph.successors(start_node):
        # If the starting molecules are not reachable from this reaction child, abort the search
        if (
            node_to_solvable[rxn_child]
            and node_to_reachable_starting_mols[rxn_child] >= starting_mols
        ):
            # Also abort search if any grandchildren are forbidden
            grandchildren = list(graph.successors(rxn_child))
            if not any(gc in forbidden_nodes for gc in grandchildren):
                # Main recurisve call: we partition K molecules among N children and check whether
                # each child is solvable with its allocated molecules.
                for start_mol_partition in split_into_subsets(
                    list(starting_mols), len(grandchildren)
                ):
                    for gc, allocated_start_mols in zip(grandchildren, start_mol_partition):
                        assert isinstance(gc, OrNode)
                        if not _is_route_with_starting_mols(
                            graph=graph,
                            start_node=gc,
                            starting_mols=set(allocated_start_mols),
                            forbidden_nodes=forbidden_nodes | {start_node, rxn_child},
                            node_to_solvable=node_to_solvable,
                            node_to_reachable_starting_mols=node_to_reachable_starting_mols,
                        ):
                            break
                    else:  # i.e. loop finished without breaking
                        return True

    # If the method has not returned at this point then there is no route
    return False
