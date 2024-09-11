from typing import Optional

from rdkit import Chem

from syntheseus import Bag, Molecule, Reaction
from syntheseus.interface.molecule import SMILES_SEPARATOR

ATOM_MAPPING_PROP_NAME = "molAtomMapNumber"


def remove_atom_mapping_from_mol(mol: Chem.Mol) -> None:
    """Removed the atom mapping from an rdkit molecule modifying it in place."""
    for atom in mol.GetAtoms():
        atom.ClearProp(ATOM_MAPPING_PROP_NAME)


def remove_atom_mapping(smiles: str) -> str:
    """Removes the atom mapping from a SMILES string.

    Args:
        smiles: Molecule SMILES to be modified.

    Returns:
        str: Input SMILES with atom map numbers stripped away.
    """
    mol = Chem.MolFromSmiles(smiles)
    remove_atom_mapping_from_mol(mol)

    return Chem.MolToSmiles(mol)


def remove_stereo_information(mol: Molecule) -> Molecule:
    return Molecule(Chem.MolToSmiles(mol.rdkit_mol, isomericSmiles=False))


def remove_stereo_information_from_reaction(reaction: Reaction) -> Reaction:
    return Reaction(
        reactants=Bag([remove_stereo_information(mol) for mol in reaction.reactants]),
        products=Bag([remove_stereo_information(mol) for mol in reaction.products]),
        identifier=reaction.identifier,
        metadata=reaction.metadata,
    )


def molecule_bag_from_smiles_strict(smiles: str) -> Bag[Molecule]:
    return Bag([Molecule(component) for component in smiles.split(SMILES_SEPARATOR)])


def molecule_bag_from_smiles(smiles: str) -> Optional[Bag[Molecule]]:
    try:
        return molecule_bag_from_smiles_strict(smiles)
    except ValueError:
        # If any of the components ends up invalid we return `None` instead.
        return None
