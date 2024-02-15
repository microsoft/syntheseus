"""
Tests for AND/OR graph/nodes.

Because a lot of the behaviour is implicitly tested when the algorithms are tested,
the tests here are sparse and mainly check edge cases which won't come up in algorithms.
"""
import pytest

from syntheseus.interface.bag import Bag
from syntheseus.interface.molecule import Molecule
from syntheseus.interface.reaction import SingleProductReaction
from syntheseus.search.graph.and_or import AndNode, AndOrGraph, OrNode
from syntheseus.search.graph.route import SynthesisGraph
from syntheseus.tests.search.graph.test_base import BaseNodeTest


class TestAndNode(BaseNodeTest):
    def get_node(self):
        return AndNode(
            reaction=SingleProductReaction(product=Molecule("CC"), reactants=Bag([Molecule("C")]))
        )

    def test_nodes_not_frozen(self):
        node = self.get_node()
        node.reaction = None


class TestOrNode(BaseNodeTest):
    def get_node(self):
        return OrNode(mol=Molecule("CC"))

    def test_nodes_not_frozen(self):
        node = self.get_node()
        node.mol = Molecule("CCC")


class TestAndOrGraph:
    def test_basic_properties1(self, cocs_mol: Molecule, andor_graph_minimal: AndOrGraph) -> None:
        """Test some basic properties (len, contains) on a graph."""
        assert len(andor_graph_minimal) == 7
        assert andor_graph_minimal.root_node in andor_graph_minimal
        assert andor_graph_minimal.root_mol == cocs_mol

    def test_basic_properties2(self, cocs_mol: Molecule, andor_tree_minimal: AndOrGraph) -> None:
        """Test some basic properties (len, contains) on another graph."""
        assert len(andor_tree_minimal) == 10
        assert andor_tree_minimal.root_node in andor_tree_minimal
        assert andor_tree_minimal.root_mol == cocs_mol

    def test_is_minimal_positive1(
        self, andor_graph_minimal: AndOrGraph, minimal_synthesis_graph: SynthesisGraph
    ) -> None:
        """
        Test that a minimal AND/OR graph (with loops) is correctly identified as minimal,
        and returns the right synthesis graph.
        """
        assert andor_graph_minimal.is_minimal()
        assert not andor_graph_minimal.is_tree()  # is a DAG
        assert andor_graph_minimal.to_synthesis_graph() == minimal_synthesis_graph

    def test_is_minimal_positive2(
        self, andor_tree_minimal: AndOrGraph, minimal_synthesis_graph: SynthesisGraph
    ) -> None:
        """
        Test that a minimal AND/OR graph (without loops) is correctly identified as minimal,
        and returns the right synthesis graph.
        """
        assert andor_tree_minimal.is_minimal()
        assert andor_tree_minimal.is_tree()
        assert andor_tree_minimal.to_synthesis_graph() == minimal_synthesis_graph

    def is_minimal_negative1(self, andor_graph_non_minimal: AndOrGraph) -> None:
        """
        Test than an AND/OR graph which contains >1 route is not identified as minimal,
        and CANNOT be converted into a synthesis graph.
        """
        assert not andor_graph_non_minimal.is_minimal()
        assert not andor_graph_non_minimal.is_tree()
        with pytest.raises(AssertionError):
            andor_graph_non_minimal.to_synthesis_graph()

    def is_minimal_negative2(self, andor_tree_non_minimal: AndOrGraph) -> None:
        """Similar test as above but for a tree."""
        assert not andor_tree_non_minimal.is_minimal()
        assert andor_tree_non_minimal.is_tree()
        with pytest.raises(AssertionError):
            andor_tree_non_minimal.to_synthesis_graph()

    @pytest.mark.parametrize(
        "reason", ["root_has_parent", "unexpanded_expanded", "reactions_dont_match"]
    )
    def test_assert_validity_negative(
        self, andor_graph_non_minimal: AndOrGraph, reason: str
    ) -> None:
        """
        Test that an invalid AND/OR graph is correctly identified as invalid.

        Different reasons for invalidity are tested.
        """
        if reason == "root_has_parent":
            random_node = [
                n
                for n in andor_graph_non_minimal.nodes()
                if n is not andor_graph_non_minimal.root_node
            ][0]
            andor_graph_non_minimal._graph.add_edge(random_node, andor_graph_non_minimal.root_node)
        elif reason == "unexpanded_expanded":
            andor_graph_non_minimal.root_node.is_expanded = False
        elif reason == "reactions_dont_match":
            first_and_node: AndNode = list(  # type: ignore # doesn't understand OrNode children are always AndNode
                andor_graph_non_minimal.successors(andor_graph_non_minimal.root_node)
            )[
                0
            ]
            first_and_node.reaction = SingleProductReaction(
                product=Molecule("CCC"), reactants=first_and_node.reaction.reactants
            )

            # Not only should the graph not be valid below,
            # but specifically the reaction should also be invalid
            with pytest.raises(AssertionError):
                andor_graph_non_minimal._assert_valid_reactions()

        else:
            raise ValueError(f"Unsupported reason: {reason}")

        with pytest.raises(AssertionError):
            andor_graph_non_minimal.assert_validity()

    @pytest.mark.parametrize(
        "reason", ["root_as_reactant", "ensure_tree", "already_expanded", "wrong_product"]
    )
    def test_invalid_expansions(
        self,
        andor_graph_non_minimal: AndOrGraph,
        bad_rxn_cc_from_cocs: SingleProductReaction,
        rxn_cs_from_cc: SingleProductReaction,
        reason: str,
    ) -> None:
        """
        Test that invalid expansions raise an error.
        Note that valid expansions are tested implicitly elsewhere.
        """
        cc_nodes = [
            n
            for n in andor_graph_non_minimal.nodes()
            if isinstance(n, OrNode) and n.mol == Molecule("CC")
        ]

        if reason == "root_as_reactant":
            with pytest.raises(AssertionError):
                andor_graph_non_minimal.expand_with_reactions(
                    reactions=[bad_rxn_cc_from_cocs], node=cc_nodes[0], ensure_tree=False
                )
        elif reason == "ensure_tree":
            with pytest.raises(AssertionError):
                andor_graph_non_minimal.expand_with_reactions(
                    reactions=[
                        SingleProductReaction(
                            product=Molecule("CC"), reactants=Bag([Molecule("CO")])
                        )
                    ],
                    node=cc_nodes[0],
                    ensure_tree=True,
                )
        elif reason == "already_expanded":
            with pytest.raises(AssertionError):
                andor_graph_non_minimal.expand_with_reactions(
                    reactions=[], node=andor_graph_non_minimal.root_node, ensure_tree=False
                )
        elif reason == "wrong_product":
            with pytest.raises(AssertionError):
                andor_graph_non_minimal.expand_with_reactions(
                    reactions=[rxn_cs_from_cc], node=cc_nodes[0], ensure_tree=False
                )
        else:
            raise ValueError(f"Unsupported reason: {reason}")
