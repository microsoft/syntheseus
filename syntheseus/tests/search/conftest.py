"""Defines common fixtures for most search tests."""
from __future__ import annotations

import collections
from dataclasses import dataclass, field

import pytest

from syntheseus.interface.bag import Bag
from syntheseus.interface.models import BackwardReactionModel
from syntheseus.interface.molecule import Molecule
from syntheseus.interface.reaction import SingleProductReaction
from syntheseus.reaction_prediction.inference.toy_models import (
    LinearMoleculesToyModel,
    ListOfReactionsToyModel,
)
from syntheseus.search.algorithms.breadth_first import (
    AndOr_BreadthFirstSearch,
    MolSet_BreadthFirstSearch,
)
from syntheseus.search.graph.and_or import AndOrGraph
from syntheseus.search.graph.molset import MolSetGraph
from syntheseus.search.graph.route import SynthesisGraph
from syntheseus.search.mol_inventory import BaseMolInventory, SmilesListInventory


@dataclass
class RetrosynthesisTask:
    """Object to hold all the parts of a retrosynthesis problem. Used for testing."""

    target_mol: Molecule
    reaction_model: BackwardReactionModel
    inventory: BaseMolInventory
    known_routes: dict[str, SynthesisGraph] = field(default_factory=dict)
    incorrect_routes: dict[str, SynthesisGraph] = field(default_factory=dict)


@pytest.fixture
def rxn_cs_from_co() -> SingleProductReaction:
    return SingleProductReaction(product=Molecule("CS"), reactants=Bag([Molecule("CO")]))


@pytest.fixture
def rxn_cocs_from_coco(cocs_mol: Molecule) -> SingleProductReaction:
    return SingleProductReaction(product=cocs_mol, reactants=Bag([Molecule("COCO")]))


@pytest.fixture
def rxn_cocc_from_co_cc() -> SingleProductReaction:
    return SingleProductReaction(
        product=Molecule("COCC"), reactants=Bag([Molecule("CO"), Molecule("CC")])
    )


@pytest.fixture
def rxn_co_from_cc() -> SingleProductReaction:
    return SingleProductReaction(product=Molecule("CO"), reactants=Bag([Molecule("CC")]))


@pytest.fixture
def rxn_cocc_from_coco() -> SingleProductReaction:
    return SingleProductReaction(product=Molecule("COCC"), reactants=Bag([Molecule("COCO")]))


@pytest.fixture
def rxn_coco_from_co() -> SingleProductReaction:
    return SingleProductReaction(product=Molecule("COCO"), reactants=Bag([Molecule("CO")]))


@pytest.fixture
def bad_rxn_cc_from_cocs() -> SingleProductReaction:
    """
    A reaction which is not possible with the LinearMolecules model,
    and has the root node of most tests as a reactant.
    Used to test illegal expansions.
    """
    return SingleProductReaction(product=Molecule("CC"), reactants=Bag([Molecule("COCS")]))


@pytest.fixture
def retrosynthesis_task1(
    cocs_mol: Molecule,
    rxn_cocs_from_coco: SingleProductReaction,
    rxn_coco_from_co: SingleProductReaction,
) -> RetrosynthesisTask:
    """Returns a retrosynthesis task which can be solved in a single step."""

    # Object for best route
    best_route = SynthesisGraph(
        SingleProductReaction(product=cocs_mol, reactants=Bag([Molecule("CO"), Molecule("CS")]))
    )

    # Object for route COCS -> COCO ; COCO -> CO
    other_route = SynthesisGraph(rxn_cocs_from_coco)
    other_route._graph.add_edge(other_route.root_node, rxn_coco_from_co)
    other_route.assert_validity()

    return RetrosynthesisTask(
        target_mol=cocs_mol,
        reaction_model=LinearMoleculesToyModel(use_cache=True),
        inventory=SmilesListInventory(["CO", "CS"]),
        known_routes={"min-cost": best_route, "other": other_route},
    )


@pytest.fixture
def retrosynthesis_task2(
    cocs_mol: Molecule,
    rxn_cocs_from_co_cs: SingleProductReaction,
    rxn_cs_from_cc: SingleProductReaction,
    rxn_cs_from_co: SingleProductReaction,
    rxn_cocs_from_cocc: SingleProductReaction,
    rxn_cocc_from_co_cc: SingleProductReaction,
    rxn_cocc_from_coco: SingleProductReaction,
    rxn_coco_from_co: SingleProductReaction,
) -> RetrosynthesisTask:
    """
    Returns a retrosynthesis task which requires 2 steps to solve.

    A 2 step solution is:
    COCS -> CO + CS
    CS -> CC
    """

    # Create various reaction objects
    known_routes: dict[str, SynthesisGraph] = dict()
    incorrect_routes: dict[str, SynthesisGraph] = dict()

    # Create some optimal shortest-length routes
    # Object for COCS -> CO + CS ; CS -> CC
    best_route = SynthesisGraph(rxn_cocs_from_co_cs)
    best_route._graph.add_edge(best_route.root_node, rxn_cs_from_cc)
    known_routes["min-cost1"] = best_route
    del best_route

    # Object for COCS -> COCC ; COCC -> CO + CC (another candidate for the best route)
    best_route2 = SynthesisGraph(rxn_cocs_from_cocc)
    best_route2._graph.add_edge(rxn_cocs_from_cocc, rxn_cocc_from_co_cc)
    known_routes["min-cost2"] = best_route2
    del best_route2

    # Object for COCS -> CO + CS ; CS -> CO (another best route)
    best_route3 = SynthesisGraph(rxn_cocs_from_co_cs)
    best_route3._graph.add_edge(rxn_cocs_from_co_cs, rxn_cs_from_co)
    known_routes["min-cost3"] = best_route3
    del best_route3

    # Create some sub-optimal routes (> 2 reactions) which should still be found
    # COCS -> COCC ; COCC -> COCO ; COCO -> CO
    other_route1 = SynthesisGraph(rxn_cocs_from_cocc)
    other_route1._graph.add_edge(rxn_cocs_from_cocc, rxn_cocc_from_coco)
    other_route1._graph.add_edge(rxn_cocc_from_coco, rxn_coco_from_co)
    known_routes["other1"] = other_route1
    del other_route1

    # Create some known incorrect routes
    incorrect_routes["1"] = SynthesisGraph(
        rxn_cocs_from_co_cs
    )  # incorrect because doesn't end in purchasable molecules
    incorrect_routes["2"] = SynthesisGraph(
        SingleProductReaction(product=cocs_mol, reactants=Bag([Molecule("COCSCOCS")]))
    )  # incorrect because this reaction cannot be output by this reaction model

    # Check that all routes are valid
    for r in known_routes.values():
        r.assert_validity()

    for r in incorrect_routes.values():
        r.assert_validity()

    return RetrosynthesisTask(
        target_mol=cocs_mol,
        reaction_model=LinearMoleculesToyModel(use_cache=True),
        inventory=SmilesListInventory(["CO", "CC"]),
        known_routes=known_routes,
        incorrect_routes=incorrect_routes,
    )


@pytest.fixture
def retrosynthesis_task3(cocs_mol: Molecule) -> RetrosynthesisTask:
    """
    Returns a retrosynthesis task which requires at least 3 steps to solve.

    The 3 step solution is:
    COCS -> CO + CS
    CS -> CC
    CO -> CC
    """
    return RetrosynthesisTask(
        target_mol=cocs_mol,
        reaction_model=LinearMoleculesToyModel(use_cache=True),
        inventory=SmilesListInventory(["CC"]),
    )


@pytest.fixture
def retrosynthesis_task4(cocs_mol: Molecule) -> RetrosynthesisTask:
    """
    A small, *finite* retrosynthesis task which can be solved in 3 steps.
    It should be easy to solve this task by completely expanding the tree.

    The 3 step solution is:
    COCS -> CO + CS
    C -> C + S
    CO -> C + O
    """
    return RetrosynthesisTask(
        target_mol=cocs_mol,
        reaction_model=LinearMoleculesToyModel(allow_substitution=False, use_cache=True),
        inventory=SmilesListInventory(["C", "O", "S"]),
    )


@pytest.fixture
def retrosynthesis_task5() -> RetrosynthesisTask:
    """
    A very small, *infinite* retrosynthesis task which cannot be solved in 1 step.
    Good for testing full expansion of trees.
    """

    return RetrosynthesisTask(
        target_mol=Molecule("CC", make_rdkit_mol=False),
        reaction_model=LinearMoleculesToyModel(allow_substitution=True, use_cache=True),
        inventory=SmilesListInventory(["O"]),
    )


@pytest.fixture
def retrosynthesis_task6() -> RetrosynthesisTask:
    """
    A large *infinite* retrosynthesis task which can generate many routes.
    Useful for testing long searches and route extraction.

    There is one single-reaction route:
    CCCOC -> CC + COC

    There are 2 routes of length 2:
    CCCOC -> CCCO + C
    C -> O

    CCCOC -> CCCOO
    CCCOO -> CCCO + O

    From route extraction, there appear to be 8 routes of length 3.
    """
    return RetrosynthesisTask(
        target_mol=Molecule("CCCOC", make_rdkit_mol=False),
        reaction_model=LinearMoleculesToyModel(allow_substitution=True, use_cache=True),
        inventory=SmilesListInventory(["CCCO", "CC", "COC", "O"]),
    )


@pytest.fixture
def rxn_model_for_minimal_graphs(
    rxn_cocs_from_co_cs: SingleProductReaction,
    rxn_co_from_cc: SingleProductReaction,
    rxn_cs_from_co: SingleProductReaction,
) -> ListOfReactionsToyModel:
    """
    Return a reaction model to help build the minimal graphs.
    Contains the following reactions:

    COCS -> CO + CS
    CO -> CC
    CS -> CO
    """
    return ListOfReactionsToyModel(
        reaction_list=[rxn_cocs_from_co_cs, rxn_co_from_cc, rxn_cs_from_co],
        use_cache=True,
    )


@pytest.fixture
def rxn_model_for_non_minimal_graphs(
    rxn_model_for_minimal_graphs: ListOfReactionsToyModel,
    rxn_cocs_from_cocc: SingleProductReaction,
    rxn_cocc_from_co_cc: SingleProductReaction,
) -> BackwardReactionModel:
    """Add reactions COCS -> COCC ; COCC -> CO + CC to the reaction model above."""
    rxn_model_for_minimal_graphs.reaction_list.extend([rxn_cocs_from_cocc, rxn_cocc_from_co_cc])
    return rxn_model_for_minimal_graphs


@pytest.fixture
def inventory_for_graph_tests() -> BaseMolInventory:
    """Return a reaction model to help build the minimal graphs."""
    return SmilesListInventory(["CC"])


def _complete_andor_graph_with_bfs(
    mol: Molecule, rxn_model: BackwardReactionModel, inventory: BaseMolInventory, unique_nodes: bool
) -> AndOrGraph:
    bfs = AndOr_BreadthFirstSearch(
        mol_inventory=inventory,
        reaction_model=rxn_model,
        limit_iterations=1000,
        unique_nodes=unique_nodes,
    )
    output_graph, _ = bfs.run_from_mol(mol)
    output_graph.assert_validity()
    return output_graph


def _complete_molset_graph_with_bfs(
    mol: Molecule, rxn_model: BackwardReactionModel, inventory: BaseMolInventory
) -> MolSetGraph:
    bfs = MolSet_BreadthFirstSearch(
        mol_inventory=inventory, reaction_model=rxn_model, limit_iterations=1000, unique_nodes=False
    )
    output_graph, _ = bfs.run_from_mol(mol)
    output_graph.assert_validity()
    return output_graph


@pytest.fixture
def minimal_synthesis_graph(
    rxn_cocs_from_co_cs: SingleProductReaction,
    rxn_co_from_cc: SingleProductReaction,
    rxn_cs_from_co: SingleProductReaction,
) -> SynthesisGraph:
    """
    Returns the synthesis graph for the minimal routes below.
    """
    g = SynthesisGraph(rxn_cocs_from_co_cs)
    g._graph.add_edge(rxn_cocs_from_co_cs, rxn_co_from_cc)
    g._graph.add_edge(rxn_cocs_from_co_cs, rxn_cs_from_co)
    g._graph.add_edge(rxn_cs_from_co, rxn_co_from_cc)
    return g


@pytest.fixture
def andor_graph_minimal(
    cocs_mol: Molecule,
    rxn_model_for_minimal_graphs: BackwardReactionModel,
    inventory_for_graph_tests: BaseMolInventory,
) -> AndOrGraph:
    """Return a minimal AND/OR *graph* with reactions from above."""
    return _complete_andor_graph_with_bfs(
        cocs_mol, rxn_model_for_minimal_graphs, inventory_for_graph_tests, True
    )


@pytest.fixture
def andor_tree_minimal(
    cocs_mol: Molecule,
    rxn_model_for_minimal_graphs: BackwardReactionModel,
    inventory_for_graph_tests: BaseMolInventory,
) -> AndOrGraph:
    """
    Return a *tree* version of the fixture above.
    """
    return _complete_andor_graph_with_bfs(
        cocs_mol, rxn_model_for_minimal_graphs, inventory_for_graph_tests, False
    )


@pytest.fixture
def andor_graph_non_minimal(
    cocs_mol: Molecule,
    rxn_model_for_non_minimal_graphs: BackwardReactionModel,
    inventory_for_graph_tests: BaseMolInventory,
) -> AndOrGraph:
    r"""
    Return a graph based on `andor_graph_minimal` with extra reactions.

    Looks like:

                    COCS
               /              \
        rxn:COCS->CO+CS     rxn:COCS->COCC
         /      \                    \
         |      CS                  COCC
         |       |                    |
         |    rxn:CS->CO       rxn:COCC->CO+CC
          \      |           /          |
           ---- CO-----------           |
               |                        |
            rxn:CO->CC                  |
                      \ _______________CC
    """

    return _complete_andor_graph_with_bfs(
        cocs_mol, rxn_model_for_non_minimal_graphs, inventory_for_graph_tests, True
    )


@pytest.fixture
def andor_tree_non_minimal(
    cocs_mol: Molecule,
    rxn_model_for_non_minimal_graphs: BackwardReactionModel,
    inventory_for_graph_tests: BaseMolInventory,
) -> AndOrGraph:
    """Tree version of above."""

    return _complete_andor_graph_with_bfs(
        cocs_mol, rxn_model_for_non_minimal_graphs, inventory_for_graph_tests, False
    )


@pytest.fixture
def molset_tree_non_minimal(
    cocs_mol: Molecule,
    rxn_model_for_non_minimal_graphs: BackwardReactionModel,
    inventory_for_graph_tests: BaseMolInventory,
) -> MolSetGraph:
    """A non-minimal MolSet tree."""

    return _complete_molset_graph_with_bfs(
        mol=cocs_mol,
        rxn_model=rxn_model_for_non_minimal_graphs,
        inventory=inventory_for_graph_tests,
    )


@pytest.fixture
def molset_tree_almost_minimal(
    cocs_mol: Molecule,
    rxn_model_for_minimal_graphs: BackwardReactionModel,
    inventory_for_graph_tests: BaseMolInventory,
) -> MolSetGraph:
    """
    A non-minimal MolSet tree which is *almost* minimal:
    it contains only reactions which could form a minimal tree,
    but because they can be executed in different orders it will
    not be minimal.
    """

    return _complete_molset_graph_with_bfs(
        mol=cocs_mol, rxn_model=rxn_model_for_minimal_graphs, inventory=inventory_for_graph_tests
    )


@pytest.fixture
def molset_tree_minimal(molset_tree_almost_minimal: MolSetGraph) -> MolSetGraph:
    """
    A minimal MolSetGraph formed by removing reactions from the almost minimal graph
    to make it minimal.
    """

    # Remove the reaction CO -> CC from the almost minimal graph + all successors
    nodes_to_remove = collections.deque(
        [
            node
            for node in molset_tree_almost_minimal.nodes()
            if node.mols == {Molecule("CC"), Molecule("CS")}
        ]
    )
    while len(nodes_to_remove) > 0:
        n = nodes_to_remove.popleft()
        for successor in molset_tree_almost_minimal.successors(n):
            nodes_to_remove.append(successor)
        molset_tree_almost_minimal._graph.remove_node(n)

    molset_tree_almost_minimal.assert_validity()
    return molset_tree_almost_minimal
