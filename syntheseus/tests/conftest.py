from __future__ import annotations

import pytest

from syntheseus.interface.bag import Bag
from syntheseus.interface.molecule import Molecule
from syntheseus.interface.reaction import SingleProductReaction


@pytest.fixture
def cocs_mol() -> Molecule:
    """Returns the molecule with smiles 'COCS'."""
    return Molecule("COCS", make_rdkit_mol=False)


@pytest.fixture
def rxn_cocs_from_co_cs(cocs_mol: Molecule) -> SingleProductReaction:
    """Returns a reaction with COCS as the product."""
    return SingleProductReaction(product=cocs_mol, reactants=Bag([Molecule("CO"), Molecule("CS")]))


@pytest.fixture
def rxn_cs_from_cc() -> SingleProductReaction:
    return SingleProductReaction(product=Molecule("CS"), reactants=Bag([Molecule("CC")]))


@pytest.fixture
def rxn_cocs_from_cocc(cocs_mol: Molecule) -> SingleProductReaction:
    return SingleProductReaction(product=cocs_mol, reactants=Bag([Molecule("COCC")]))
