"""Code related to starting molecules match metric (called exact set-wise match in FusionRetro)."""

from __future__ import annotations

import itertools
import typing
from collections.abc import Iterable

from syntheseus.search.chem import Molecule
from syntheseus.search.graph.and_or import ANDOR_NODE, AndOrGraph, OrNode

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
        _starting_mols_under_each_node(graph),
    )


def partition_set(A: list[T], k: int) -> Iterable[list[list[T]]]:
    """
    Enumerates all possible ways to partition a list A into k disjoint subsets which are non-empty
    (in all possible orders).

    This function is easiest to explain by example:

    A single partition is just the list A itself.

    >>> list(partition_set([1, 2, 3], 1))
    >>> [[[1, 2, 3]]]

    For k=2, there are 3 possible partitions each with two possible orders.

    >>> list(partition_set([1, 2, 3], 2))
    >>> [[[1], [2, 3]], [[2], [1, 3]], [[3], [1, 2]], [[1, 2], [3]], [[1, 3], [2]], [[2, 3], [1]]]

    For k=3, there is only 1 possible partition but 6 possible orders.

    >>> list(partition_set([1, 2, 3], 3))
    >>> [[[1], [2], [3]], [[1], [3], [2]], [[2], [1], [3]], [[2], [3], [1]], [[3], [1], [2]], [[3], [2], [1]]]

    This function uses a recursive implementation.
    First it partitions A into 2 (where the second partition has at least k-1 elements),
    then it recursively partitions the second partition into (k-1) partitions.
    """

    # Check 1: elements of list are unique
    assert len(set(A)) == len(A)

    # Check 2: k is valid
    assert k >= 1

    # Base case 1: list is empty
    if len(A) == 0:
        return

    # Base case 2: just a single partition
    if k == 1:
        yield [A]
        return

    # Main case: partition A into two parts, then recursively partition the second part
    max_size_of_first_partition = len(A) - k + 1
    for first_partition_size in range(1, max_size_of_first_partition + 1):
        for first_partition in itertools.combinations(A, first_partition_size):
            # Find the remaining elements to partition.
            # NOTE: this assumes that the elements of A are unique.
            remaining_elements = [x for x in A if x not in first_partition]
            for subsequent_partitions in partition_set(remaining_elements, k - 1):
                yield [list(first_partition)] + subsequent_partitions


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
    node_to_all_reachable_starting_mols: dict[ANDOR_NODE, set[Molecule]],
) -> bool:
    """
    Recursive method to check whether there is a route in the graph,
    starting from `start_node` and excluding `forbidden_nodes`,
    whose leaves and exactly `starting_mols`.

    To prune the search, we use the `node_to_all_reachable_starting_mols` dictionary,
    which contains the set of all purchasable molecules reachable under each node (not necessarily part of a single route though).
    We use this to prune the search early: if some molecules cannot be reached at all, there is no point checking whether
    they might be reachable from a single route.
    """
    assert start_node in graph

    # Base case 1: starting mols is empty
    if len(starting_mols) == 0:
        return False

    # Base case 2: start node is forbidden
    if start_node in forbidden_nodes:
        return False

    # Base case 3: starting are not reachable at all from the start node
    if not (starting_mols <= node_to_all_reachable_starting_mols[start_node]):
        return False

    # Base case 4: there is just one starting molecule and this OrNode contains it.
    if len(starting_mols) == 1 and list(starting_mols)[0] == start_node.mol:
        return True

    # Main case: the required starting molecules are reachable,
    # but we just need to check whether they are reachable within a single synthesis route.
    # We do this by explicitly trying to find this synthesis route.
    for rxn_child in graph.successors(start_node):
        # If the starting molecules are not reachable from this reaction child, abort the search
        if node_to_all_reachable_starting_mols[rxn_child] >= starting_mols:
            # Also abort search if any grandchildren are forbidden
            grandchildren = list(graph.successors(rxn_child))
            if not any(gc in forbidden_nodes for gc in grandchildren):
                # Main recurisve call: we partition K molecules among N children and check whether
                for start_mol_partition in partition_set(list(starting_mols), len(grandchildren)):
                    for gc, allocated_start_mols in zip(grandchildren, start_mol_partition):
                        assert isinstance(gc, OrNode)
                        if not _is_route_with_starting_mols(
                            graph,
                            gc,
                            set(allocated_start_mols),
                            forbidden_nodes | {start_node, rxn_child},
                            node_to_all_reachable_starting_mols,
                        ):
                            break
                    else:  # i.e. loop finished without breaking
                        return True

    # If the method has not returned at this point then there is no route
    return False
