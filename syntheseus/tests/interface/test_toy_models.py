import pytest

from syntheseus.interface.molecule import Molecule
from syntheseus.interface.reaction import SingleProductReaction
from syntheseus.reaction_prediction.inference import (
    LinearMoleculesToyModel,
    ListOfReactionsToyModel,
)


def test_linear_molecules_invalid_molecule() -> None:
    """
    The LinearMolecules model is only defined on the space of linear molecules.
    If called on a non-linear molecule (e.g. with branching) it is currently set up to throw an error.
    This test ensures that this happens.

    NOTE: in the future the behaviour could be changed to just return an empty list,
    but for a toy example I thought it would be best to alert the user with a warning.
    """
    rxn_model = LinearMoleculesToyModel()
    with pytest.raises(AssertionError):
        rxn_model([Molecule("CC(C)C")])


def test_list_of_reactions_model(
    rxn_cocs_from_co_cs: SingleProductReaction,
    rxn_cocs_from_cocc: SingleProductReaction,
    rxn_cs_from_cc: SingleProductReaction,
) -> None:
    """Simple test of the ListOfReactionsModel class."""
    model = ListOfReactionsToyModel([rxn_cocs_from_co_cs, rxn_cocs_from_cocc, rxn_cs_from_cc])
    output = model([Molecule("COCS"), Molecule("CS"), Molecule("CO")])
    assert output == [[rxn_cocs_from_co_cs, rxn_cocs_from_cocc], [rxn_cs_from_cc], []]
