"""
Classes to hold molecules, without reference to the reactions they may take part in.
"""

from dataclasses import InitVar, dataclass, field
from typing import Optional, Union

from rdkit import Chem

from syntheseus.interface.bag import Bag
from syntheseus.interface.typed_dict import TypedDict

SMILES_SEPARATOR = "."


class MoleculeMetaData(TypedDict, total=False):
    """Class to add typing to optional meta-data fields for molecules."""

    rdkit_mol: Chem.Mol

    # Things related to multi-step retrosynthesis
    is_purchasable: bool
    cost: float
    supplier: str

    # Other potentially relevant data
    purity: float


@dataclass(frozen=True, order=True)
class Molecule:
    """
    Object representing a molecule with its SMILES string and an optional
    identifier to distinguish molecules with identical SMILES strings (usually not used).
    Everything else is considered metadata and is stored in a dictionary which is not
    compared or hashed.

    The class is frozen since it should not need to be edited,
    and this will auto-implement __eq__ and __hash__ methods.

    On initialization it is possible to automatically convert to canonical
    smiles (default True), and to store the rdkit molecule (default True).
    If set to false, there is no guarantee of canonicalization or storage of
    an rdkit mol.
    """

    smiles: str = field(hash=True, compare=True)
    identifier: Optional[Union[str, int]] = field(default=None, hash=True, compare=True)

    canonicalize: InitVar[bool] = True
    make_rdkit_mol: InitVar[bool] = True

    metadata: MoleculeMetaData = field(
        default_factory=lambda: MoleculeMetaData(),
        hash=False,
        compare=False,
    )

    def __post_init__(self, canonicalize: bool, make_rdkit_mol: bool) -> None:
        if canonicalize or make_rdkit_mol:
            try:
                rdkit_mol = Chem.MolFromSmiles(self.smiles)
            except Exception as e:
                raise ValueError(f"Cannot create a molecule with SMILES '{self.smiles}'") from e

            if make_rdkit_mol:
                self.metadata["rdkit_mol"] = rdkit_mol

            if canonicalize:
                try:
                    smiles_canonical = Chem.MolToSmiles(rdkit_mol)
                except Exception as e:
                    raise ValueError(
                        f"Cannot canonicalize a molecule with SMILES '{self.smiles}'"
                    ) from e

                object.__setattr__(self, "smiles", smiles_canonical)

    @property
    def rdkit_mol(self) -> Chem.Mol:
        """Makes an rdkit mol if one does yet exist"""
        if "rdkit_mol" not in self.metadata:
            self.metadata["rdkit_mol"] = Chem.MolFromSmiles(self.smiles)
        return self.metadata["rdkit_mol"]


def molecule_bag_to_smiles(mols: Bag[Molecule]) -> str:
    """Combine SMILES strings of molecules in a `Bag` into a single string.

    For two bags that represent the same multiset of molecules this function will return the same
    result, because iteration order over a `Bag` is deterministic (sorted using default comparator).
    """
    return SMILES_SEPARATOR.join(mol.smiles for mol in mols)
