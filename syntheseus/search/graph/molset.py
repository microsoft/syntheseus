from __future__ import annotations

import datetime
import warnings
from collections import Counter
from collections.abc import Collection
from dataclasses import dataclass
from typing import Optional, Sequence

import networkx as nx

from syntheseus.interface.molecule import Molecule
from syntheseus.interface.reaction import SingleProductReaction
from syntheseus.search.graph.base_graph import RetrosynthesisSearchGraph
from syntheseus.search.graph.node import BaseGraphNode
from syntheseus.search.graph.route import SynthesisGraph


@dataclass(eq=False)
class _BaseMolSetNode:
    """
    This class exists due to restrictions on inheriting from dataclasses.
    E.g. https://stackoverflow.com/questions/51575931/class-inheritance-in-python-3-7-dataclasses
    """

    mols: frozenset[Molecule]


@dataclass(eq=False)
class MolSetNode(BaseGraphNode, _BaseMolSetNode):
    """
    A node whose state corresponds to a set of molecules.
    Used for the "OR tree" formulation of retrosynthesis
    as an MDP where the state is a set of molecules
    and actions are reactions which remove some molecules from the set
    and add in others.
    """

    def _has_intrinsic_solution(self) -> bool:
        return all(mol.metadata["is_purchasable"] for mol in self.mols)

    def _has_solution_from_children(self, children: Collection["MolSetNode"]) -> bool:  # type: ignore[override]
        return any(n.has_solution for n in children)


class MolSetGraph(RetrosynthesisSearchGraph[MolSetNode]):
    """Search graph where all nodes represent sets of molecules, and edges are annotated with reactions."""

    def __init__(
        self,
        *args,
        root_node: MolSetNode,
        one_node_per_molset: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self._root_node = root_node
        self._graph.add_node(root_node)
        self._one_node_per_molset = one_node_per_molset
        self._molset_to_node: dict[frozenset[Molecule], MolSetNode] = dict()
        if self._one_node_per_molset:
            warnings.warn(
                "Using one_node_per_molset=True is not well-tested so is not recommended."
                " In a future version it will either be removed or better-supported."
            )
            self._molset_to_node[self._root_node.mols] = self._root_node

    @property
    def root_node(self) -> MolSetNode:
        return self._root_node

    @property
    def root_mol(self) -> Molecule:
        root_mols = self.root_node.mols
        assert len(root_mols) == 1
        return list(root_mols)[0]

    def is_minimal(self) -> bool:
        # Has a single route if each node has just one child except for one final leaf node
        for node in self._graph.nodes:
            if len(list(self.successors(node))) > 1:
                return False

        return nx.is_weakly_connected(self._graph)

    def _assert_valid_reactions(self) -> None:
        # Check reactants and products match
        for node in self._graph.nodes:
            assert isinstance(node, MolSetNode)
            for child_node in self.successors(node):
                edge_data = self._graph.get_edge_data(node, child_node)
                assert "reaction" in edge_data
                rxn = edge_data["reaction"]
                assert ((set(node.mols) - {rxn.product}) | rxn.unique_reactants) == child_node.mols

    def expand_with_reactions(
        self,
        reactions: list[SingleProductReaction],
        node: MolSetNode,
        ensure_tree: bool,
    ) -> Sequence[MolSetNode]:
        if not ensure_tree:
            warnings.warn("ensure_tree=False is not well-tested so be careful!")

        # Check that parent is in the graph already
        assert node in self

        # Check that reactions are acceptable
        assert all(
            r.product in node.mols for r in reactions
        ), "All reactions must have a product in 'node'."

        # Check whether it is already expanded.
        # NOTE: behaviour could change in the future to just yield a warning
        assert not node.is_expanded

        # Create and add nodes one at a time
        creation_time = datetime.datetime.now(datetime.timezone.utc)
        new_nodes: list[MolSetNode] = list()
        node.is_expanded = True
        for reaction in reactions:
            new_mol_set = frozenset(
                (set(node.mols) - {reaction.product}) | reaction.unique_reactants
            )
            if new_mol_set in self._molset_to_node:
                new_node = self._molset_to_node[new_mol_set]
            else:
                new_node = MolSetNode(
                    mols=new_mol_set,
                    creation_time=creation_time,
                )
            assert new_node is not self.root_node, "Root node should not be a child."
            self._graph.add_node(new_node)
            new_nodes.append(new_node)
            self._graph.add_edge(node, new_node, reaction=reaction)

            if ensure_tree:
                assert len(list(self.predecessors(new_node))) == 1

            # Optionally update mol -> node map
            if self._one_node_per_molset:
                self._molset_to_node[new_mol_set] = new_node

        return list(dict.fromkeys(new_nodes))

    def to_synthesis_graph(self, nodes: Optional[Collection[MolSetNode]] = None) -> SynthesisGraph:
        wrong_child_error_str = (
            "The root node should have exactly one child to extract a synthesis graph."
        )

        # Choose subgraph
        if nodes is None:
            subgraph = self._graph
        else:
            assert self.root_node in nodes
            subgraph = self._graph.subgraph(nodes)

        # Find root reaction to initialize graph
        root_successors = list(subgraph.successors(self.root_node))
        assert len(root_successors) == 1, wrong_child_error_str
        curr_successor = root_successors[0]
        root_rxn = subgraph.edges[self.root_node, curr_successor]["reaction"]
        new_graph = SynthesisGraph(root_node=root_rxn)

        # Keep track of molecules without reactions
        mol_to_reactions_with_mol = {mol: [root_rxn] for mol in root_rxn.reactants}
        while True:
            # Find next reaction
            curr_successors = list(subgraph.successors(curr_successor))
            if len(curr_successors) == 0:
                break  # We've reached the end of the graph
            assert len(curr_successors) == 1, wrong_child_error_str
            next_successor = curr_successors[0]
            next_rxn = subgraph.edges[curr_successor, next_successor]["reaction"]

            # Update current successor
            curr_successor = next_successor

            # Add reaction to graph
            new_graph._graph.add_node(next_rxn)
            for parent_rxn in mol_to_reactions_with_mol.pop(next_rxn.product):
                new_graph._graph.add_edge(parent_rxn, next_rxn)

            # Update mols without reactions
            for prod_mol in next_rxn.reactants:
                mol_to_reactions_with_mol.setdefault(prod_mol, []).append(next_rxn)

        # Check for validity
        new_graph.assert_validity()

        return new_graph

    def smiles_set_counter(self) -> Counter[tuple[str, ...]]:
        """
        Returns a Counter object with the number of times each set of SMILES strings
        appears in the graph.
        Useful for testing.
        """
        all_smiles_sets = [tuple(sorted(mol.smiles for mol in node.mols)) for node in self.nodes()]
        return Counter(all_smiles_sets)
