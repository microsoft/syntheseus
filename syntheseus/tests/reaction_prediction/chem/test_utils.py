from rdkit import Chem

from syntheseus import Bag, Molecule, Reaction, SingleProductReaction
from syntheseus.reaction_prediction.chem.utils import (
    remove_atom_mapping,
    remove_atom_mapping_from_mol,
    remove_stereo_information,
    remove_stereo_information_from_reaction,
)


def test_remove_mapping() -> None:
    smiles_mapped = "[OH:1][CH2:2][c:3]1[cH:4][n:5][cH:6][cH:7][c:8]1[Br:9]"
    smiles_unmapped = "OCc1cnccc1Br"

    assert remove_atom_mapping(smiles_mapped) == smiles_unmapped
    assert remove_atom_mapping(smiles_unmapped) == smiles_unmapped

    mol = Chem.MolFromSmiles(smiles_mapped)
    remove_atom_mapping_from_mol(mol)

    assert Chem.MolToSmiles(mol) == smiles_unmapped


def test_remove_stereo_information() -> None:
    mol = Molecule("CC(N)C#N")
    mols_chiral = [Molecule("C[C@H](N)C#N"), Molecule("C[C@@H](N)C#N")]

    assert len(set([mol] + mols_chiral)) == 3
    assert len(set([mol] + [remove_stereo_information(m) for m in mols_chiral])) == 1


def test_remove_stereo_information_from_reaction() -> None:
    reactants = Bag([Molecule("CCC"), Molecule("CC(N)C#N")])
    reactants_chiral = Bag([Molecule("CCC"), Molecule("C[C@H](N)C#N")])

    product = Molecule("CC(N)C#N")
    product_chiral = Molecule("C[C@H](N)C#N")

    rxn = Reaction(reactants=reactants, products=Bag([product]))
    rxn_chiral = Reaction(reactants=reactants_chiral, products=Bag([product_chiral]))
    rxn_stereo_removed = remove_stereo_information_from_reaction(rxn_chiral)

    assert type(rxn_stereo_removed) is Reaction
    assert rxn_stereo_removed == rxn

    sp_rxn = SingleProductReaction(reactants=reactants, product=product)
    sp_rxn_chiral = SingleProductReaction(reactants=reactants_chiral, product=product_chiral)
    sp_rxn_stero_removed = remove_stereo_information_from_reaction(sp_rxn_chiral)

    assert type(sp_rxn_stero_removed) is SingleProductReaction
    assert sp_rxn_stero_removed == sp_rxn
