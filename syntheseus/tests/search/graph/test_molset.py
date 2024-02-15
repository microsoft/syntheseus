"""
Tests for MolSet graph/nodes.

Because a lot of the behaviour is implicitly tested when the algorithms are tested,
the tests here are sparse and mainly check edge cases which won't come up in algorithms.
"""
import pytest

from syntheseus.interface.bag import Bag
from syntheseus.interface.molecule import Molecule
from syntheseus.interface.reaction import SingleProductReaction
from syntheseus.search.graph.molset import MolSetGraph, MolSetNode
from syntheseus.search.graph.route import SynthesisGraph
from syntheseus.tests.search.graph.test_base import BaseNodeTest


class TestMolSetNode(BaseNodeTest):
    def get_node(self):
        return MolSetNode(mols=Bag([Molecule("CC")]))

    def test_nodes_not_frozen(self):
        node = self.get_node()
        node.mols = None


class TestMolSetGraph:
    """Tests for MolSetGraph: they mostly follow the same pattern as AND/OR graph tests."""

    def test_basic_properties1(
        self, cocs_mol: Molecule, molset_tree_non_minimal: MolSetGraph
    ) -> None:
        """Test some basic properties (len, contains) on a graph."""
        assert len(molset_tree_non_minimal) == 10
        assert molset_tree_non_minimal.root_node in molset_tree_non_minimal
        assert molset_tree_non_minimal.root_mol == cocs_mol

    def _check_not_minimal(self, graph: MolSetGraph) -> None:
        """Checks that a graph is not minimal (eliminate code duplication between tests below)."""
        assert not graph.is_minimal()
        assert graph.is_tree()  # should always be a tree
        with pytest.raises(AssertionError):
            graph.to_synthesis_graph()

    def test_minimal_negative(self, molset_tree_non_minimal: MolSetGraph) -> None:
        """Test that a non-minimal molset graph is not identified as minimal."""
        self._check_not_minimal(molset_tree_non_minimal)

    def test_minimal_negative_hard(self, molset_tree_almost_minimal: MolSetGraph) -> None:
        """Harder test for minimal negative: the 'almost minimal' graph."""
        self._check_not_minimal(molset_tree_almost_minimal)

    def test_minimal_positive(
        self, molset_tree_minimal: MolSetGraph, minimal_synthesis_graph: SynthesisGraph
    ) -> None:
        graph = molset_tree_minimal
        assert graph.is_minimal()
        assert graph.is_tree()  # should always be a tree
        route = graph.to_synthesis_graph()  # should run without error
        assert route == minimal_synthesis_graph

    @pytest.mark.parametrize(
        "reason", ["root_has_parent", "unexpanded_expanded", "reactions_dont_match"]
    )
    def test_assert_validity_negative(
        self, molset_tree_non_minimal: MolSetGraph, reason: str
    ) -> None:
        """
        Test that an invalid MolSet graph is correctly identified as invalid.

        Different reasons for invalidity are tested.
        """
        graph = molset_tree_non_minimal

        if reason == "root_has_parent":
            # Add a random connection to the root node
            random_node = [n for n in graph.nodes() if n is not graph.root_node][0]
            graph._graph.add_edge(random_node, graph.root_node)
        elif reason == "unexpanded_expanded":
            graph.root_node.is_expanded = False
        elif reason == "reactions_dont_match":
            child = list(graph.successors(graph.root_node))[0]

            # Set the reaction to a random incorrect reaction
            graph._graph.edges[graph.root_node, child]["reaction"] = SingleProductReaction(
                reactants=Bag([Molecule("OO")]), product=Molecule("CC")
            )

            # Not only should the graph not be valid below,
            # but specifically the reaction should also be invalid
            with pytest.raises(AssertionError):
                graph._assert_valid_reactions()
        else:
            raise ValueError(f"Unsupported reason: {reason}")

        with pytest.raises(AssertionError):
            graph.assert_validity()

    @pytest.mark.parametrize("reason", ["already_expanded", "wrong_product"])
    def test_invalid_expansions(
        self,
        molset_tree_non_minimal: MolSetGraph,
        rxn_cs_from_cc: SingleProductReaction,
        reason: str,
    ) -> None:
        """
        Test that invalid expansions raise an error.
        Note that valid expansions are tested implicitly elsewhere.

        NOTE: because graphs with 1 node per molset are not properly supported now,
        we don't test the case where the product of a reaction is the root mol.
        """
        graph = molset_tree_non_minimal
        cc_node = [n for n in graph.nodes() if n.mols == {Molecule("CC")}].pop()

        if reason == "already_expanded":
            with pytest.raises(AssertionError):
                graph.expand_with_reactions(reactions=[], node=graph.root_node, ensure_tree=True)
        elif reason == "wrong_product":
            with pytest.raises(AssertionError):
                graph.expand_with_reactions(
                    reactions=[rxn_cs_from_cc], node=cc_node, ensure_tree=True
                )
        else:
            raise ValueError(f"Unsupported reason: {reason}")
