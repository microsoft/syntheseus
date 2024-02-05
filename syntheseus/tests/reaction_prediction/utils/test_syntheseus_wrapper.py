from typing import List, Sequence

from syntheseus.interface.bag import Bag
from syntheseus.interface.models import BackwardReactionModel
from syntheseus.interface.molecule import Molecule
from syntheseus.interface.reaction import SingleProductReaction
from syntheseus.reaction_prediction.utils.syntheseus_wrapper import (
    SyntheseusBackwardReactionModel,
)


class MockBackwardReactionModel(BackwardReactionModel):
    """For every molecule, return a reaction that converts it to itself."""

    def __call__(
        self, inputs: List[Molecule], num_results: int
    ) -> List[Sequence[SingleProductReaction]]:
        return [[SingleProductReaction(product=mol, reactants=Bag([mol]))] for mol in inputs]

    def get_parameters(self):
        return []


def test_syntheseus_wrapper() -> None:
    model = MockBackwardReactionModel()
    syntheseus_model = SyntheseusBackwardReactionModel(model=model, num_results=100)
    mols = [Molecule(smiles="CC"), Molecule(smiles="CCC")]
    output = syntheseus_model(mols=mols)

    assert output == [
        [
            SingleProductReaction(
                product=Molecule(smiles="CC"), reactants=Bag([Molecule(smiles="CC")])
            ),
        ],
        [
            SingleProductReaction(
                product=Molecule(smiles="CCC"), reactants=Bag([Molecule(smiles="CCC")])
            ),
        ],
    ]
