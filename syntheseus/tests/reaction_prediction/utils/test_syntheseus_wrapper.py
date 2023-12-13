from typing import List

from syntheseus.interface.bag import Bag
from syntheseus.interface.models import (
    BackwardReactionModel,
    Prediction,
    PredictionList,
)
from syntheseus.interface.molecule import Molecule
from syntheseus.reaction_prediction.utils.syntheseus_wrapper import (
    SyntheseusBackwardReactionModel,
)
from syntheseus.search.chem import BackwardReaction


class MockBackwardReactionModel(BackwardReactionModel):
    """For every molecule, return a reaction that converts it to itself."""

    def __call__(
        self, inputs: List[Molecule], num_results: int
    ) -> List[PredictionList[Molecule, Bag[Molecule]]]:
        return [
            PredictionList(input=mol, predictions=[Prediction(input=mol, output=Bag([mol]))])
            for mol in inputs
        ]

    def get_parameters(self):
        return []


def test_syntheseus_wrapper() -> None:
    model = MockBackwardReactionModel()
    syntheseus_model = SyntheseusBackwardReactionModel(model=model, num_results=100)
    mols = [Molecule(smiles="CC"), Molecule(smiles="CCC")]
    output = syntheseus_model(mols=mols)

    assert output == [
        [
            BackwardReaction(
                product=Molecule(smiles="CC"), reactants=frozenset([Molecule(smiles="CC")])
            ),
        ],
        [
            BackwardReaction(
                product=Molecule(smiles="CCC"), reactants=frozenset([Molecule(smiles="CCC")])
            ),
        ],
    ]
