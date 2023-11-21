from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from typing import Union

from syntheseus.search.chem import BackwardReaction, Molecule
from syntheseus.search.graph.base_graph import BaseReactionGraph

MOL_AND_RXN = Union[Molecule, BackwardReaction]


class SynthesisGraph(BaseReactionGraph[BackwardReaction]):
    """
    Data structure used to hold a retrosynthesis graph containing only
    reaction objects. The purpose of this class is as a minimal container
    for route objects, instead of storing them as AndOrGraphs or MolSetGraphs.
    """

    def __init__(self, root_node: BackwardReaction, **kwargs) -> None:
        super().__init__(**kwargs)
        self._root_node = root_node
        self._graph.add_node(self._root_node)

    @property
    def root_node(self) -> BackwardReaction:
        return self._root_node

    @property
    def root_mol(self) -> Molecule:
        return self.root_node.product

    def is_minimal(self) -> bool:
        # Check if any product appears more than once
        for rxn in self._graph.nodes:
            product_count = Counter([rxn.product for rxn in self.successors(rxn)])
            if any(v > 1 for v in product_count.values()):
                return False
        return True

    def assert_validity(self) -> None:
        # Everything from superclass applies
        super().assert_validity()

        for node in self._graph.nodes:
            assert isinstance(node, BackwardReaction)
            for parent in self.predecessors(node):
                assert isinstance(parent, BackwardReaction)
                assert node.product in parent.reactants
            children = list(self.successors(node))
            assert len(children) == len(set(children))  # all children should be unique
            assert set([child.product for child in children]) <= set(
                node.reactants
            )  # all children should be reactants

    def expand_with_reactions(
        self,
        reactions: list[BackwardReaction],
        node: BackwardReaction,
        ensure_tree: bool,
    ) -> Sequence[BackwardReaction]:
        raise NotImplementedError

    def get_starting_molecules(self) -> set[Molecule]:
        """
        Get the 'starting molecules' for this route,
        i.e. reactant molecules which are not a product of a child reaction.
        """
        output: set[Molecule] = set()
        for rxn in self._graph.nodes:
            successor_products = {child_rxn.product for child_rxn in self.successors(rxn)}
            for reactant in rxn.reactants:
                if reactant not in successor_products:
                    output.add(reactant)
        return output

    def __str__(self) -> str:
        return str([rxn.reaction_smiles for rxn in self.nodes()])
