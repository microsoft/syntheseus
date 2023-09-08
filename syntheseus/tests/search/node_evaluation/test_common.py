import math

import pytest

from syntheseus.search.graph.and_or import AndNode, AndOrGraph
from syntheseus.search.graph.molset import MolSetGraph
from syntheseus.search.node_evaluation.common import (
    ConstantNodeEvaluator,
    HasSolutionValueFunction,
    ReactionModelLogProbCost,
    ReactionModelProbPolicy,
)


class TestConstantNodeEvaluator:
    @pytest.mark.parametrize("constant", [0.3, 0.8])
    def test_values(self, constant: float, andor_graph_non_minimal: AndOrGraph) -> None:
        val_fn = ConstantNodeEvaluator(constant)
        vals = val_fn(list(andor_graph_non_minimal.nodes()))
        assert all([v == constant for v in vals])  # values should match
        assert val_fn.num_calls == len(
            andor_graph_non_minimal
        )  # should have been called once per node


class TestHasSolutionValueFunction:
    def test_values(self, andor_graph_non_minimal: AndOrGraph) -> None:
        val_fn = HasSolutionValueFunction()
        nodes = list(andor_graph_non_minimal.nodes())
        vals = val_fn(nodes)
        assert all([v == float(n.has_solution) for v, n in zip(vals, nodes)])  # values should match
        assert val_fn.num_calls == len(
            andor_graph_non_minimal
        )  # should have been called once per node


class TestReactionModelLogProbCost:
    @pytest.mark.parametrize("normalize", [False, True])
    @pytest.mark.parametrize("temperature", [1.0, 2.0])
    @pytest.mark.parametrize("clip_probability_min", [0.1, 0.5])
    @pytest.mark.parametrize("clip_probability_max", [0.5, 1.0])
    def test_values(
        self,
        andor_graph_non_minimal: AndOrGraph,
        normalize: bool,
        temperature: float,
        clip_probability_min: float,
        clip_probability_max: float,
    ) -> None:
        val_fn = ReactionModelLogProbCost(
            normalize=normalize,
            temperature=temperature,
            clip_probability_min=clip_probability_min,
            clip_probability_max=clip_probability_max,
        )
        nodes = [node for node in andor_graph_non_minimal.nodes() if isinstance(node, AndNode)]

        # The toy model does not set reaction probabilities, so set these manually.
        node_val_expected = {}
        for idx, node in enumerate(nodes):
            prob = idx / (len(nodes) - 1)
            node.reaction.metadata["probability"] = prob  # type: ignore

            node_val_expected[node] = (
                -math.log(min(clip_probability_max, max(clip_probability_min, prob))) / temperature
            )

        if normalize:
            normalization_constant = math.log(sum(math.exp(-v) for v in node_val_expected.values()))
            node_val_expected = {
                key: value + normalization_constant for key, value in node_val_expected.items()
            }

        vals = val_fn(nodes)
        for val_computed, node in zip(vals, nodes):  # values should match
            assert math.isclose(val_computed, node_val_expected[node])

        assert val_fn.num_calls == len(nodes)  # should have been called once per AND node

    def test_enforces_min_clipping(self) -> None:
        with pytest.raises(ValueError):
            ReactionModelLogProbCost(clip_probability_min=0.0)  # should fail as `return_log = True`


class TestReactionModelProbPolicy:
    @pytest.mark.parametrize("normalize", [False, True])
    @pytest.mark.parametrize("temperature", [1.0, 2.0])
    @pytest.mark.parametrize("clip_probability_min", [0.1, 0.5])
    @pytest.mark.parametrize("clip_probability_max", [0.5, 1.0])
    def test_values(
        self,
        molset_tree_non_minimal: MolSetGraph,
        normalize: bool,
        temperature: float,
        clip_probability_min: float,
        clip_probability_max: float,
    ) -> None:
        val_fn = ReactionModelProbPolicy(
            normalize=normalize,
            temperature=temperature,
            clip_probability_min=clip_probability_min,
            clip_probability_max=clip_probability_max,
        )
        nodes = [
            node
            for node in molset_tree_non_minimal.nodes()
            if node != molset_tree_non_minimal.root_node
        ]

        # The toy model does not set reaction probabilities, so set these manually.
        node_val_expected = {}
        for idx, node in enumerate(nodes):
            [parent] = molset_tree_non_minimal.predecessors(node)
            reaction = molset_tree_non_minimal._graph.edges[parent, node]["reaction"]

            # Be careful not to overwrite things as some reactions in the graph are repeated.
            if "probability" not in reaction.metadata:
                reaction.metadata["probability"] = prob = idx / (len(nodes) - 1)
            else:
                prob = reaction.metadata["probability"]

            node_val_expected[node] = min(
                clip_probability_max, max(clip_probability_min, prob)
            ) ** (1.0 / temperature)

        if normalize:
            normalization_factor = sum(node_val_expected.values())
            node_val_expected = {
                key: value / normalization_factor for key, value in node_val_expected.items()
            }

        vals = val_fn(nodes, graph=molset_tree_non_minimal)
        for val_computed, node in zip(vals, nodes):  # values should match
            assert math.isclose(val_computed, node_val_expected[node])

        assert (
            val_fn.num_calls == len(molset_tree_non_minimal) - 1
        )  # should have been called once per non-root node

    def test_enforces_min_clipping(self) -> None:
        with pytest.raises(ValueError):
            ReactionModelProbPolicy(clip_probability_min=0.0)  # should fail as `normalize = True`

        ReactionModelProbPolicy(
            normalize=False, clip_probability_min=0.0
        )  # should succeed if we explicitly turn off normalization
