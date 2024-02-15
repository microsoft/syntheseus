from __future__ import annotations

import abc
from collections.abc import Container, Iterable, Sized
from typing import Generic, Sequence, TypeVar

import networkx as nx

from syntheseus.interface.molecule import Molecule
from syntheseus.interface.reaction import SingleProductReaction
from syntheseus.search.graph.node import BaseGraphNode

NodeType = TypeVar(
    "NodeType",
)
SearchNodeType = TypeVar("SearchNodeType", bound=BaseGraphNode)


class BaseReactionGraph(Container, Sized, Generic[NodeType], abc.ABC):
    """
    Base class for holding a retrosynthesis graph where nodes represent molecules/reactions.
    Retrosynthesis graphs have the following properties:

    - Directed: A node usually represents a reaction/molecule used to synthesize its predecessors.
        Search generally goes from predecessors to successors,
        while the actual synthesis would go from successors to predecessors.
    - Root node: there is a root node representing the molecule to be synthesized.
        This node should never have a parent.
    - Implicit: the graph is a subgraph of a much larger reaction graph,
        so nodes can be "unexpanded", meaning that no children nodes are specified,
        even though the node should have children. Search typically "expands"
        a node by adding reactions.

    The actual implementation of the graph uses networkx's DiGraph
    (but it does not inherit from it because this causes many methods to break).
    """

    def __init__(self, *args, **kwargs) -> None:
        self._graph = nx.DiGraph(*args, **kwargs)

    def __contains__(self, node: object) -> bool:
        return node in self._graph

    def __len__(self) -> int:
        return len(self._graph)

    @property
    @abc.abstractmethod
    def root_node(self) -> NodeType:
        """Root node of the graph, representing the molecule to be synthesized."""
        pass

    @property
    @abc.abstractmethod
    def root_mol(self) -> Molecule:
        """The molecule to be synthesized."""
        pass

    @abc.abstractmethod
    def is_minimal(self) -> bool:
        """Checks whether this is a *minimal* graph (i.e. contains a single synthesis route)."""
        pass

    def is_tree(self) -> bool:
        """Performs a [possibly expensive] check to see if the graph is a tree."""
        return nx.is_arborescence(self._graph)

    def nodes(self) -> Iterable[NodeType]:
        return self._graph.nodes()

    def predecessors(self, node: NodeType) -> Iterable[NodeType]:
        """Returns the predecessors of a node."""
        return self._graph.predecessors(node)

    def successors(self, node: NodeType) -> Iterable[NodeType]:
        """Returns the successors of a node."""
        return self._graph.successors(node)

    def assert_validity(self) -> None:
        """
        A (potentially expensive) function to check the graph's validity.
        """

        # Check root node is in the graph
        assert self.root_node in self

        # Check root node has no parents
        assert len(list(self.predecessors(self.root_node))) == 0

        # Graph should be connected
        assert nx.is_weakly_connected(self._graph)

    def __eq__(self, __value: object) -> bool:
        """Equality is defined as having the same root node, nodes, and edges."""
        if isinstance(__value, BaseReactionGraph):
            return (self.root_node == __value.root_node) and nx.utils.graphs_equal(
                self._graph, __value._graph
            )
        else:
            return False

    @abc.abstractmethod
    def expand_with_reactions(
        self,
        reactions: list[SingleProductReaction],
        node: NodeType,
        ensure_tree: bool,
    ) -> Sequence[NodeType]:
        """
        Expands a node with a series of reactions.
        For reproducibility, it ensures that the order of the nodes is consistent,
        which is why a sequence is returned.
        It is also encouraged (but not required) for all return nodes to be unique.

        Subclass implementations should ensure that the root node never has any predecessors
        as a result of this function, and if ensure_tree=True, should also ensure that the
        graph remains a tree.
        """
        pass


class RetrosynthesisSearchGraph(BaseReactionGraph[SearchNodeType], Generic[SearchNodeType]):
    """Subclass with more specific type requirements for the nodes."""

    @abc.abstractmethod
    def _assert_valid_reactions(self) -> None:
        """Checks that all reactions are valid."""
        pass

    def assert_validity(self) -> None:
        super().assert_validity()

        # Check that all nodes with children are marked as expanded
        for n in self._graph.nodes:
            if not n.is_expanded:
                assert len(list(self.successors(n))) == 0

        # Check valid reactions
        self._assert_valid_reactions()
