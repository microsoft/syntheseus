from __future__ import annotations

import math
import warnings
from typing import cast

from syntheseus.interface.molecule import Molecule
from syntheseus.interface.reaction import SingleProductReaction
from syntheseus.search.graph.and_or import ANDOR_NODE, AndNode, AndOrGraph, OrNode
from syntheseus.search.graph.base_graph import RetrosynthesisSearchGraph
from syntheseus.search.graph.message_passing import (
    depth_update,
    has_solution_update,
    run_message_passing,
)
from syntheseus.search.graph.molset import MolSetGraph, MolSetNode


def _make_unique_node_andor_graph(
    root_mol: Molecule,
    mol_to_node: dict[Molecule, OrNode],
    rxn_to_node: dict[SingleProductReaction, AndNode],
) -> AndOrGraph:
    # Make new graph
    new_graph = AndOrGraph(root_node=mol_to_node[root_mol], one_node_per_molecule=True)

    # Add all nodes
    new_graph._graph.add_nodes_from(mol_to_node.values())
    new_graph._graph.add_nodes_from(rxn_to_node.values())
    new_graph._mol_to_node.update(mol_to_node)

    # Add all edges
    for rxn, and_node in rxn_to_node.items():
        new_graph._graph.add_edge(mol_to_node[rxn.product], and_node)
        for reactant in rxn.unique_reactants:
            new_graph._graph.add_edge(and_node, mol_to_node[reactant])

        # Mark relevant nodes as expanded
        mol_to_node[rxn.product].is_expanded = True
        and_node.is_expanded = True

    # Validate graph (should be valid at this point)
    new_graph.assert_validity()

    # Run update functions
    all_new_nodes = list(new_graph._graph.nodes())
    run_message_passing(  # depth
        new_graph, all_new_nodes, update_fns=[depth_update], update_predecessors=False
    )
    run_message_passing(  # has_solution
        new_graph,
        sorted(all_new_nodes, key=lambda node: node.depth, reverse=True),  # for efficiency
        update_fns=[has_solution_update],
        update_successors=False,
    )

    return new_graph


def _unique_node_andor_from_andor(
    graph: AndOrGraph,
) -> AndOrGraph:
    # Get all mols and reactions
    all_nodes: set[ANDOR_NODE] = set(graph._graph.nodes())
    mols = set(node.mol for node in all_nodes if isinstance(node, OrNode))
    rxns = set(node.reaction for node in all_nodes if isinstance(node, AndNode))

    # Make initial nodes
    mol_to_node = {mol: OrNode(mol) for mol in mols}
    rxn_to_node = {rxn: AndNode(rxn) for rxn in rxns}

    # Transfer time attributes
    for node in all_nodes:
        # Get relevant node
        if isinstance(node, OrNode):
            node_in_new_graph: ANDOR_NODE = mol_to_node[node.mol]
        elif isinstance(node, AndNode):
            node_in_new_graph = rxn_to_node[node.reaction]
        else:
            raise TypeError(f"Unknown node type: {type(node)}")

        # Set time attributes
        node_in_new_graph.creation_time = min(node_in_new_graph.creation_time, node.creation_time)
        for k in ["num_calls_rxn_model", "num_calls_value_function"]:
            if k in node.data:
                # doesn't understand typeddict
                node_in_new_graph.data[k] = min(  # type: ignore[literal-required]
                    cast(int, node_in_new_graph.data.get(k, math.inf)),
                    cast(int, node.data[k]),  # type: ignore[literal-required]
                )

    return _make_unique_node_andor_graph(
        root_mol=graph.root_mol, mol_to_node=mol_to_node, rxn_to_node=rxn_to_node
    )


def _unique_node_andor_from_molset(
    graph: MolSetGraph,
) -> AndOrGraph:
    # Right now this only definitely works for trees
    if not graph.is_tree():
        warnings.warn(
            "The outputs of this function are only guaranteed to be correct for trees.",
            category=UserWarning,
        )

    # Get all mols and reactions
    all_nodes: set[MolSetNode] = set(graph._graph.nodes())
    mols = set(mol for node in all_nodes for mol in node.mols)
    rxns = set(graph._graph.edges[n1, n2]["reaction"] for n1, n2 in graph._graph.edges())

    # Make initial nodes
    mol_to_node = {mol: OrNode(mol) for mol in mols}
    rxn_to_node = {rxn: AndNode(rxn) for rxn in rxns}

    # Transfer time attributes
    for node in all_nodes:
        # Mol nodes: this node's times are candidate for the minimum
        for mol in node.mols:
            mol_to_node[mol].creation_time = min(mol_to_node[mol].creation_time, node.creation_time)

            for k in ["num_calls_rxn_model", "num_calls_value_function"]:
                if k in node.data:
                    mol_to_node[mol].data[k] = min(  # type: ignore[literal-required]
                        cast(int, mol_to_node[mol].data.get(k, math.inf)),
                        cast(int, node.data[k]),  # type: ignore[literal-required]
                    )

        # Reaction nodes: the MAX of (this node, parent node) is a candidate for the minimum
        # (although this may not be recovered exactly)
        for parent in graph.predecessors(node):
            rxn = graph._graph.edges[parent, node]["reaction"]

            # Creation time
            c_time = max(node.creation_time, parent.creation_time)
            if c_time < rxn_to_node[rxn].creation_time:
                rxn_to_node[rxn].creation_time = c_time

            for k in ["num_calls_rxn_model", "num_calls_value_function"]:
                if k in node.data:
                    cand_time = max(node.data[k], parent.data[k])  # type: ignore[literal-required]
                    rxn_to_node[rxn].data[k] = min(  # type: ignore[literal-required]
                        rxn_to_node[rxn].data.get(k, math.inf),
                        cand_time,
                    )

    return _make_unique_node_andor_graph(
        root_mol=graph.root_mol, mol_to_node=mol_to_node, rxn_to_node=rxn_to_node
    )


def get_unique_node_andor_graph(graph: RetrosynthesisSearchGraph) -> AndOrGraph:
    r"""
    Given a search graph, use the reactions in the graph to create a new AndOrGraph
    with exactly one node per unique reaction and one node per unique molecule. In
    effect, this compacts the graph to reduce duplicate nodes. For example, an
    AndOrGraph like:

                             A
                            /  \
                           B    C
                          / \  / \
                         D  E  E  F


    would be converted to:
                             A
                            / \
                           B   C
                          / \ / \
                         D   E   F
    """
    warnings.warn("This function is not well-tested: use with caution.")
    if isinstance(graph, AndOrGraph):
        return _unique_node_andor_from_andor(graph)
    elif isinstance(graph, MolSetGraph):
        return _unique_node_andor_from_molset(graph)
    else:
        raise NotImplementedError(f"Cannot convert {type(graph)} to AndOrGraph")
