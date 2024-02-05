from syntheseus.interface.bag import Bag
from syntheseus.interface.molecule import Molecule
from syntheseus.interface.reaction import MultiProductReaction, SingleProductReaction

# TODO(@austint): more tests will follow once this class is integrated into the library
# because tests from prior reaction classes will be ported here


def test_reaction_objects():
    """Instantiate a `Reaction` object."""
    C2 = Molecule(2 * "C")
    C3 = Molecule(3 * "C")
    C5 = Molecule(5 * "C")

    # Single product reaction
    rxn1 = SingleProductReaction(
        reactants=Bag([C2, C3]),
        product=C5,
    )
    assert rxn1.reaction_smiles == "CC.CCC>>CCCCC"

    # Multi-product reaction
    rxn2 = MultiProductReaction(
        reactants=Bag([C5]),
        product=Bag([C2, C3]),
    )
    assert rxn2.reaction_smiles == "CCCCC>>CC.CCC"
