from __future__ import annotations

import abc
import math
import random
import warnings
from collections.abc import Collection
from datetime import datetime
from typing import Generic, Optional, Sequence, TypeVar

from syntheseus.interface.models import BackwardReactionModel
from syntheseus.interface.molecule import Molecule
from syntheseus.interface.reaction import SingleProductReaction
from syntheseus.search import INT_INF
from syntheseus.search.graph.and_or import AndOrGraph, OrNode
from syntheseus.search.graph.base_graph import RetrosynthesisSearchGraph
from syntheseus.search.graph.message_passing import (
    depth_update,
    has_solution_update,
    run_message_passing,
)
from syntheseus.search.graph.molset import MolSetGraph, MolSetNode
from syntheseus.search.graph.node import BaseGraphNode
from syntheseus.search.mol_inventory import BaseMolInventory

AlgReturnType = TypeVar("AlgReturnType")
GraphType = TypeVar("GraphType", bound=RetrosynthesisSearchGraph)


class MinimalSearchAlgorithm(Generic[GraphType, AlgReturnType], abc.ABC):
    """Defines minimal interface for search algorithms."""

    def __init__(
        self, reaction_model: BackwardReactionModel, mol_inventory: BaseMolInventory, **kwargs
    ):
        super().__init__(**kwargs)
        self.reaction_model = reaction_model
        self.mol_inventory = mol_inventory

    @abc.abstractmethod
    def run_from_graph(self, graph: GraphType) -> AlgReturnType:
        """Perform search on the graph. Return arbitrary information."""
        pass

    @abc.abstractmethod
    def run_from_mol(self, mol: Molecule) -> tuple[GraphType, AlgReturnType]:
        """
        Initialize a graph from a molecule and run search on it.
        This function should probably call self.run_from_graph, but this is not enforced.
        """
        pass

    def reset(self):
        """
        Reset everything for this algorithm.

        Subclasses which need to reset things should override this method,
        call super().reset(), then reset other information.
        """
        self.reaction_model.reset()


class SearchAlgorithm(MinimalSearchAlgorithm[GraphType, AlgReturnType]):
    """
    Parent class for search algorithms in this repo, which implements
    some common functions for all algorithms to reduce code duplication.
    """

    def __init__(
        self,
        limit_iterations: int = INT_INF,
        limit_reaction_model_calls: int = INT_INF,
        limit_graph_nodes: int = INT_INF,
        time_limit_s: float = math.inf,
        max_expansion_depth: int = 50,
        expand_purchasable_mols: bool = False,
        set_depth: bool = True,
        set_has_solution: bool = True,
        unique_nodes: bool = False,
        random_state: Optional[random.Random] = None,
        prevent_repeat_mol_in_trees: bool = False,
        stop_on_first_solution: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.limit_iterations = limit_iterations
        self.limit_reaction_model_calls = limit_reaction_model_calls
        self.limit_graph_nodes = limit_graph_nodes
        self.time_limit_s = time_limit_s
        self.max_expansion_depth = max_expansion_depth
        self.expand_purchasable_mols = expand_purchasable_mols
        self.set_depth = set_depth
        self.set_has_solution = set_has_solution
        self.stop_on_first_solution = stop_on_first_solution

        # Unique nodes
        if self.requires_tree and unique_nodes:
            raise ValueError("Unique nodes cannot be used since this will not give a tree.")
        self.unique_nodes = unique_nodes

        # Random state
        self.random_state = random_state or random.Random()

        # Repeat mol
        self.prevent_repeat_mol_in_trees = prevent_repeat_mol_in_trees
        if self.unique_nodes and self.prevent_repeat_mol_in_trees:
            warnings.warn(
                "prevent_repeat_mol_in_trees=True when unique_nodes=True is redundant"
                " since the graph will not be a tree and there will be no repeat mols."
            )

        # Warning about lack of caching in reaction model
        if not self.reaction_model._use_cache:
            warnings.warn(
                "The reaction model does not use caching, which may result "
                "in unnecessary duplicate calls with the same input molecule."
            )

    @property
    def requires_tree(self) -> bool:
        """Whether this algorithm must be run on a tree."""
        return True

    def run_from_mol(self, mol: Molecule) -> tuple[GraphType, AlgReturnType]:
        graph = self.create_graph(mol)
        return graph, self.run_from_graph(graph)

    @abc.abstractmethod
    def create_graph(self, mol: Molecule) -> GraphType:
        """Initialize a graph from a single molecule."""
        raise NotImplementedError

    def setup(self, graph: GraphType) -> None:
        """Does setup for the algorithm."""

        # Record some initial values
        self._start_time = datetime.now()

        # Ensure graph is valid
        graph.assert_validity()
        if self.requires_tree:
            assert graph.is_tree(), "Graph must be a tree!"

        # Set initial values for all nodes
        self.set_node_values(graph._graph.nodes(), graph)

    def teardown(self, graph: GraphType) -> None:
        """Does teardown for the algorithm."""
        del self._start_time

    def run_from_graph(self, graph: GraphType) -> AlgReturnType:
        self.setup(graph)
        output = self._run_from_graph_after_setup(graph)
        self.teardown(graph)
        return output

    @abc.abstractmethod
    def _run_from_graph_after_setup(self, graph: GraphType) -> AlgReturnType:
        """Main method for subclasses to override, which forces them to do setup and teardown."""
        raise NotImplementedError

    def should_stop_search(self, graph) -> bool:
        """
        Generic checking function for whether search should stop.

        Base implementation checks whether the time limit has been reached
        (both wall clock time and calls to the reaction model)
        and whether to stop search because a solution was found (only if `stop_on_first_solution is True`).

        Importantly, this function does NOT check whether the iteration limit is reached:
        this is because an "iteration" means different things for different algorithms.
        We recommend putting this check in the main loop of the algorithm.
        """
        elapsed_time = (
            datetime.now() - self._start_time
        ).total_seconds()  # NOTE: `self._start_time` is set in `setup`
        return (
            (elapsed_time >= self.time_limit_s)
            or (self.reaction_model.num_calls() >= self.limit_reaction_model_calls)
            or (len(graph) >= self.limit_graph_nodes)
            or (self.stop_on_first_solution and graph.root_node.has_solution)
        )

    def set_node_values(
        self, nodes: Collection[BaseGraphNode], graph: GraphType
    ) -> Collection[BaseGraphNode]:
        """
        Set any/all values for a set of nodes.
        Returns all nodes reached during the update process.

        Base implementation sets depth and "has solution" values,
        and some time variables.
        """

        # Fill some values for each node
        for node in nodes:
            # Initialize number of calls to reaction model
            node.data.setdefault("num_calls_rxn_model", self.reaction_model.num_calls())

            # Fill molecule metadata from inventory
            self._fill_molecule_metadata(node, graph)

        # Run message passing
        output_nodes = set(nodes)
        if self.set_depth:
            output_nodes.update(
                run_message_passing(
                    graph=graph,
                    nodes=output_nodes,
                    update_fns=[depth_update],
                    update_predecessors=False,
                )
            )

        if self.set_has_solution:
            output_nodes.update(
                run_message_passing(
                    graph=graph,
                    nodes=output_nodes,
                    update_fns=[has_solution_update],
                    update_successors=False,
                )
            )

        return output_nodes

    @abc.abstractmethod
    def _fill_molecule_metadata(self, node: BaseGraphNode, graph: GraphType) -> None:
        """Fills all molecule metadata."""
        raise NotImplementedError

    @abc.abstractmethod
    def _get_mols_to_expand(self, node: BaseGraphNode, graph: GraphType) -> Collection[Molecule]:
        raise NotImplementedError

    @abc.abstractmethod
    def _filter_reactions(
        self, reactions: Sequence[SingleProductReaction], node: BaseGraphNode, graph: GraphType
    ) -> list[SingleProductReaction]:
        """Remove unwanted reactions from the list."""
        raise NotImplementedError

    def can_expand_node(self, node: BaseGraphNode, graph: GraphType) -> bool:
        """
        Return whether a node is eligible to be expanded. It is used by self.expand_node below,
        but is made available so that algorithms can check whether a node *would* be expanded.

        The base implementation checks whether the node is already expanded and whether the node is beyond `max_expansion_depth`.
        Subclasses could add additional functionality.
        """
        too_deep = node.depth >= self.max_expansion_depth
        already_expanded = node.is_expanded
        no_mols = len(self._get_mols_to_expand(node, graph)) == 0
        return not (too_deep or already_expanded or no_mols)

    def expand_node(
        self, node: BaseGraphNode, graph: GraphType, force_expansion: bool = False
    ) -> Sequence[BaseGraphNode]:
        """
        In the default case, checks self.can_expand_node, and if this passes then the node is expanded with the reaction model.

        If force_expansion=True, then this check is skipped and the node is expanded regardless of whether it should be.
        """

        if force_expansion or self.can_expand_node(node, graph):
            # Get molecules to expand
            mols = list(self._get_mols_to_expand(node, graph))

            # Optionally terminate without expansion if there are no molecules to expand
            if len(mols) == 0:
                return list()

            # Get reactions for each of these molecules
            rxn_model_output = self.reaction_model(mols)

            # Filter reactions to remove unwanted ones
            filtered_rxn_list = [
                self._filter_reactions(rxn_list, node, graph) for rxn_list in rxn_model_output
            ]

            # Add new nodes to the graph
            new_nodes: list[BaseGraphNode] = list(
                graph.expand_with_reactions(
                    [rxn for rxn_list in filtered_rxn_list for rxn in rxn_list],
                    node,
                    ensure_tree=not self.unique_nodes,
                )
            )

            # Return unique nodes, but in a consistent order
            return list(dict.fromkeys(new_nodes))
        else:
            return list()


class AndOrSearchAlgorithm(SearchAlgorithm[AndOrGraph, AlgReturnType], Generic[AlgReturnType]):
    def create_graph(self, mol: Molecule) -> AndOrGraph:
        return AndOrGraph(
            root_node=OrNode(mol=mol, num_visit=0), one_node_per_molecule=self.unique_nodes
        )

    def _fill_molecule_metadata(self, node: BaseGraphNode, graph: AndOrGraph) -> None:
        if isinstance(node, OrNode):
            self.mol_inventory.fill_metadata(node.mol)

    def _get_mols_to_expand(self, node: BaseGraphNode, graph: AndOrGraph) -> Collection[Molecule]:
        output: list[Molecule] = []
        if isinstance(node, OrNode):
            if self.expand_purchasable_mols or not node.mol.metadata["is_purchasable"]:
                output.append(node.mol)
        return output

    def _get_tree_ancestor_mols(self, node: OrNode, graph: AndOrGraph) -> Collection[Molecule]:
        """
        Get ancestor molecules for this OrNode, assuming the graph is a tree,
        *including* this node's mol.
        """
        mols: set[Molecule] = {node.mol}
        curr_node = node
        while curr_node.depth > 0:
            and_predecessors = list(graph.predecessors(curr_node))
            assert len(and_predecessors) == 1, "This graph is not a tree!"
            or_predecessors = list(graph.predecessors(and_predecessors[0]))
            assert len(or_predecessors) == 1, "This graph is not a tree!"
            curr_node = or_predecessors[0]  # type: ignore
            assert isinstance(curr_node, OrNode)
            mols.add(curr_node.mol)
        return mols

    def _filter_reactions(
        self, reactions: Sequence[SingleProductReaction], node: BaseGraphNode, graph: AndOrGraph
    ) -> list[SingleProductReaction]:
        # Filter out any reactions that contain the root molecule
        reactions = [rxn for rxn in reactions if graph.root_node.mol not in rxn.reactants]

        # Optionally filter out A -> ... -> A ... cycles
        if self.prevent_repeat_mol_in_trees:
            assert isinstance(node, OrNode)
            ancestor_mols = self._get_tree_ancestor_mols(node, graph)
            reactions = [
                rxn for rxn in reactions if not any(r in ancestor_mols for r in rxn.reactants)
            ]

        return reactions


class MolSetSearchAlgorithm(SearchAlgorithm[MolSetGraph, AlgReturnType], Generic[AlgReturnType]):
    def create_graph(self, mol: Molecule) -> MolSetGraph:
        return MolSetGraph(
            root_node=MolSetNode(mols=frozenset([mol]), num_visit=0),
            one_node_per_molset=self.unique_nodes,
        )

    def _fill_molecule_metadata(self, node: BaseGraphNode, graph: MolSetGraph) -> None:
        """Fills all molecule metadata."""
        assert isinstance(node, MolSetNode)
        for mol in node.mols:
            self.mol_inventory.fill_metadata(mol)

    def _get_mols_to_expand(self, node: BaseGraphNode, graph: MolSetGraph) -> Collection[Molecule]:
        output: list[Molecule] = []
        assert isinstance(node, MolSetNode)
        for mol in node.mols:
            if self.expand_purchasable_mols or not mol.metadata["is_purchasable"]:
                output.append(mol)
        return output

    def _get_tree_ancestor_molsets(
        self, node: MolSetNode, graph: MolSetGraph
    ) -> Collection[frozenset[Molecule]]:
        """
        Get ancestor mol sets for this MolSetNode, assuming the graph is a tree,
        *including* this node's molset.
        """
        molsets: set[frozenset[Molecule]] = {node.mols}
        curr_node = node
        while curr_node.depth > 0:
            parents = list(graph.predecessors(curr_node))
            assert len(parents) == 1, "This graph is not a tree!"
            curr_node = parents[0]
            molsets.add(curr_node.mols)
        return molsets

    def _filter_reactions(
        self, reactions: Sequence[SingleProductReaction], node: BaseGraphNode, graph: MolSetGraph
    ) -> list[SingleProductReaction]:
        # Filter out any reactions that contain the root molecule
        assert len(graph.root_node.mols) == 1
        root_mol = list(graph.root_node.mols)[0]
        reactions = [rxn for rxn in reactions if root_mol not in rxn.reactants]

        # Optionally filter out A -> B -> A ... cycles
        if self.prevent_repeat_mol_in_trees:
            assert isinstance(node, MolSetNode)
            ancestor_molsets = self._get_tree_ancestor_molsets(node, graph)
            reactions = [
                rxn
                for rxn in reactions
                if frozenset((node.mols - {rxn.product}) | rxn.unique_reactants)
                not in ancestor_molsets
            ]

        return reactions
