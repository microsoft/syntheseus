from rdkit import Chem

from syntheseus.reaction_prediction.chem.utils import (
    remove_atom_mapping,
    remove_atom_mapping_from_mol,
)


def test_remove_mapping() -> None:
    smiles_mapped = "[OH:1][CH2:2][c:3]1[cH:4][n:5][cH:6][cH:7][c:8]1[Br:9]"
    smiles_unmapped = "OCc1cnccc1Br"

    assert remove_atom_mapping(smiles_mapped) == smiles_unmapped
    assert remove_atom_mapping(smiles_unmapped) == smiles_unmapped

    mol = Chem.MolFromSmiles(smiles_mapped)
    remove_atom_mapping_from_mol(mol)

    assert Chem.MolToSmiles(mol) == smiles_unmapped
