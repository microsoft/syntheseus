from __future__ import annotations

from collections.abc import Sequence

from syntheseus.search.chem import BackwardReaction, Molecule
from syntheseus.search.reaction_models.base import BackwardReactionModel


class ListOfReactionsModel(BackwardReactionModel):
    """A model which returns reactions from a pre-defined list."""

    def __init__(self, reaction_list: Sequence[BackwardReaction], **kwargs) -> None:
        super().__init__(**kwargs)
        self.reaction_list = list(reaction_list)

    def _get_backward_reactions(self, mols: list[Molecule]) -> list[list[BackwardReaction]]:
        return [[r for r in self.reaction_list if r.product == mol] for mol in mols]


class LinearMolecules(BackwardReactionModel):
    """
    A simple toy model of "reactions" on linear "ball-and-stick" molecules,
    where the possible reactions involve string cuts and substitutions.

    Molecules in this model must be formed entirely from C,S, and O atoms with single bonds.
    The reactions allowed are:
    - string cuts, e.g. "CCOC" -> "CC" + "OC" (*see note 2 below)
    - substitution of the atom on either end of the molecule: e.g. "CCOC" -> "CCOO"

    NOTE 1: molecules formed by this model are mostly unphysical and the reactions
    are not actual chemical reactions. This model is intended for testing and debugging.

    NOTE 2: all molecules are returned with canonical SMILES, so the outputs may not look
    the same as a string cut. For example, "CCOC" -> "CC" + "OC" will get canonicalized to
    "CC" + "CO" (i.e. the "O" and "C" will swap places). Fundamentally this doesn't change anything
    since "CO" and "OC" are the same molecule, but it may be confusing when debugging.
    """

    def __init__(self, allow_substitution: bool = True, **kwargs) -> None:
        super().__init__(**kwargs)
        self._allow_substitution = allow_substitution  # should not be modified after init

    def _get_single_backward_reactions(self, mol: Molecule) -> list[BackwardReaction]:
        assert set(mol.smiles) <= set("COS"), "Molecules must be formed out of C,O, and S atoms."
        assert len(mol.smiles) > 0, "Molecules must have at least 1 atom."
        output: list[BackwardReaction] = []

        # String cuts
        for cut_idx in range(1, len(mol.smiles)):
            mol1 = Molecule(mol.smiles[:cut_idx], make_rdkit_mol=False)
            mol2 = Molecule(mol.smiles[cut_idx:], make_rdkit_mol=False)
            output.append(
                BackwardReaction(
                    product=mol,
                    reactants=frozenset([mol1, mol2]),
                    metadata={"source": f"string cut at idx {cut_idx}"},
                )
            )

        # Substitutions
        if self._allow_substitution:
            for sub_idx in {0, len(mol.smiles) - 1}:  # use set in case len(mol.smiles) == 1
                for sub_atom in "COS":
                    if mol.smiles[sub_idx] == sub_atom:
                        continue
                    else:
                        new_mol = Molecule(
                            mol.smiles[:sub_idx] + sub_atom + mol.smiles[sub_idx + 1 :],
                            make_rdkit_mol=False,
                        )
                        output.append(
                            BackwardReaction(
                                product=mol,
                                reactants=frozenset([new_mol]),
                                metadata={"source": f"substitution idx {sub_idx} with {sub_atom}"},
                            ),
                        )

        return output

    def _get_backward_reactions(self, mols: list[Molecule]) -> list[list[BackwardReaction]]:
        return [self._get_single_backward_reactions(mol) for mol in mols]
