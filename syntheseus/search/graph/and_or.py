from __future__ import annotations

import datetime
from collections import Counter
from collections.abc import Collection
from dataclasses import dataclass
from typing import Optional, Sequence, Union

import networkx as nx

from syntheseus.interface.molecule import Molecule
from syntheseus.interface.reaction import SingleProductReaction
from syntheseus.search.graph.base_graph import RetrosynthesisSearchGraph
from syntheseus.search.graph.node import BaseGraphNode
from syntheseus.search.graph.route import SynthesisGraph


@dataclass(eq=False)
class _BaseOrNode:
    """
    This class exists due to restrictions on inheriting from dataclasses.
    E.g. https://stackoverflow.com/questions/51575931/class-inheritance-in-python-3-7-dataclasses
    """

    mol: Molecule


@dataclass(eq=False)
class OrNode(BaseGraphNode, _BaseOrNode):
    def _has_intrinsic_solution(self) -> bool:
        return self.mol.metadata["is_purchasable"]

    def _has_solution_from_children(self, children: Collection["AndNode"]) -> bool:  # type: ignore[override]
        return any(n.has_solution for n in children)


@dataclass(eq=False)
class _BaseAndNode:
    """
    This class exists due to restrictions on inheriting from dataclasses.
    E.g. https://stackoverflow.com/questions/51575931/class-inheritance-in-python-3-7-dataclasses
    """

    reaction: SingleProductReaction


@dataclass(eq=False)
class AndNode(BaseGraphNode, _BaseAndNode):
    def _has_intrinsic_solution(self) -> bool:
        return False

    def _has_solution_from_children(self, children: Collection[OrNode]) -> bool:  # type: ignore[override]
        return all(n.has_solution for n in children)


ANDOR_NODE = Union[AndNode, OrNode]  # convenient type for all nodes in AND/OR graph


class AndOrGraph(RetrosynthesisSearchGraph[ANDOR_NODE]):
    def __init__(
        self,
        *args,
        root_node: OrNode,
        one_node_per_molecule: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self._root_node = root_node
        self._graph.add_node(root_node)
        self._one_node_per_molecule = one_node_per_molecule
        self._mol_to_node: dict[Molecule, OrNode] = dict()
        if self._one_node_per_molecule:
            self._mol_to_node[root_node.mol] = root_node

    @property
    def root_node(self) -> OrNode:
        return self._root_node

    @property
    def root_mol(self) -> Molecule:
        return self._root_node.mol

    def is_minimal(self) -> bool:
        # An AND/OR graph is minimal if it is connected
        # and every OR node has at most 1 child
        for node in self._graph.nodes:
            if isinstance(node, OrNode):
                if len(list(self.successors(node))) > 1:
                    return False

        return nx.is_weakly_connected(self._graph)

    def _assert_valid_reactions(self) -> None:
        for node in self._graph.nodes:
            if isinstance(node, OrNode):
                # Check types only
                for child in self.successors(node):
                    assert isinstance(child, AndNode)
                for parent in self.predecessors(node):
                    assert isinstance(parent, AndNode)
            elif isinstance(node, AndNode):
                # Parents should be OrNodes whose mol is the reaction product
                for parent in self.predecessors(node):
                    assert isinstance(parent, OrNode)
                    assert parent.mol == node.reaction.product

                # Should be 1 child OrNode for each reactant
                all_children = list(self.successors(node))
                assert len(all_children) == len(node.reaction.unique_reactants), (
                    f"Should have 1 child per reactant, but found {len(all_children)} "
                    f"children for {len(node.reaction.unique_reactants)} reactants."
                )

                for child in all_children:
                    assert isinstance(child, OrNode)
                assert set(child.mol for child in all_children) == set(node.reaction.reactants)  # type: ignore  # does not understand that all children are OrNode
            else:
                raise TypeError(f"Unexpected node type: {type(node)}")

    def assert_validity(self) -> None:
        # Everything from superclass applies
        super().assert_validity()

        if self._one_node_per_molecule:
            # Check that there is actually 1 OrNode for each molecule in the graph
            assert len(self._mol_to_node) == len(
                [n for n in self._graph.nodes if isinstance(n, OrNode)]
            )
        else:
            # Check that this dictionary is empty (i.e. not used)
            assert len(self._mol_to_node) == 0

    def expand_with_reactions(  # type: ignore[override]  # because it only accepts OrNodes
        self,
        reactions: list[SingleProductReaction],
        node: OrNode,
        ensure_tree: bool,  # raises an error if something is done to make the graph no longer a tree
    ) -> Sequence[ANDOR_NODE]:
        # Check that parent is in the graph already
        assert node in self

        # Check that reactions are acceptable
        assert all(
            r.product == node.mol for r in reactions
        ), "All reactions must have the same product."

        # Check whether it is already expanded.
        # NOTE: behaviour could change in the future to just yield a warning
        assert not node.is_expanded

        # Create and add nodes for each reaction one at a time.
        # All nodes will have the same creation time.
        creation_time = datetime.datetime.now(datetime.timezone.utc)
        new_nodes: list[ANDOR_NODE] = list()
        node.is_expanded = True
        for reaction in reactions:
            # And Node
            and_node = AndNode(
                reaction=reaction,
                creation_time=creation_time,
                is_expanded=True,
            )
            self._graph.add_node(and_node)
            new_nodes.append(and_node)
            self._graph.add_edge(node, and_node)

            for reactant_mol in reaction.unique_reactants:
                if reactant_mol in self._mol_to_node:
                    or_node = self._mol_to_node[reactant_mol]
                else:
                    or_node = OrNode(
                        mol=reactant_mol,
                        creation_time=creation_time,
                    )
                assert or_node is not self.root_node, "Root node should not be a child."
                new_nodes.append(or_node)
                self._graph.add_node(or_node)
                self._graph.add_edge(and_node, or_node)

                if ensure_tree:
                    # Check here that the new node has exactly one parent
                    # (otherwise the graph is not a tree)
                    assert len(list(self.predecessors(or_node))) == 1

                # Optionally update mol -> node map
                if self._one_node_per_molecule:
                    self._mol_to_node[reactant_mol] = or_node

        # Return new nodes, but ensure they are unique and in the same order
        return list(dict.fromkeys(new_nodes))

    def to_synthesis_graph(self, nodes: Optional[Collection[ANDOR_NODE]] = None) -> SynthesisGraph:
        """
        Returns a graph composed of reactions instead of AND/OR nodes.
        """

        # Choose subgraph
        if nodes is None:
            subgraph = self._graph
        else:
            assert self.root_node in nodes
            subgraph = self._graph.subgraph(nodes)

        # Find root reaction
        root_reactions = list(subgraph.successors(self.root_node))
        assert (
            len(root_reactions) == 1
        ), f"There appears to be {len(root_reactions)} reactions for the root node (expected exactly 1)."
        new_graph = SynthesisGraph(root_node=root_reactions[0].reaction)

        # Add all nodes and edges
        for node in subgraph.nodes:
            if isinstance(node, AndNode):
                new_graph._graph.add_node(node.reaction)
                for or_child in subgraph.successors(node):
                    for and_grandchild in subgraph.successors(or_child):
                        new_graph._graph.add_edge(node.reaction, and_grandchild.reaction)

        # Check for validity
        new_graph.assert_validity()

        return new_graph

    def smiles_counter(self) -> Counter[str]:
        """
        Returns a Counter object with the number of times each unique SMILES string appears in the graph.
        Useful for testing.
        """
        all_smiles = [node.mol.smiles for node in self.nodes() if isinstance(node, OrNode)]
        return Counter(all_smiles)

    def reaction_smiles_counter(self) -> Counter[str]:
        """
        Returns a Counter object with the number of times each reaction SMILES appears in the graph.
        Useful for testing.
        """
        return Counter(
            [node.reaction.reaction_smiles for node in self.nodes() if isinstance(node, AndNode)]
        )
