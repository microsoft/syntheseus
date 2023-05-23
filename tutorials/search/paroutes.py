"""Code relating to PaRoutes model/inventory."""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import RDLogger
from rdkit.Chem import AllChem

try:
    import keras
except ImportError:
    warnings.warn("Keras not installed, PaRoutes model will not be available.")

try:
    from rdchiral.main import rdchiralRunText
except ImportError:
    warnings.warn("rdchiral not installed, PaRoutes model will not be available.")


from syntheseus.search.chem import BackwardReaction, Molecule
from syntheseus.search.mol_inventory import SmilesListInventory
from syntheseus.search.reaction_models import BackwardReactionModel

# Various files from PaRoutes
BASE_PATH = Path(__file__).parent.absolute() / "paroutes_files"

N_LIST = [1, 5]

TARGET_FILES = {n: str(BASE_PATH / f"n{n}-targets.txt") for n in N_LIST}
STOCK_FILES = {n: str(BASE_PATH / f"n{n}-stock.txt") for n in N_LIST}
UNIQUE_TEMPLATE_FILE = str(BASE_PATH / "uspto_unique_templates.csv.gz")
MODEL_DUMP_FILE = str(BASE_PATH / "uspto_keras_model.hdf5")
ROUTE_JSON_FILES = {n: str(BASE_PATH / f"n{n}-routes.json") for n in N_LIST}


# Turn off rdkit logger
lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)


def get_target_smiles(n: int) -> list[str]:
    with open(TARGET_FILES[n]) as f:
        lines = f.readlines()
    return [line.strip() for line in lines]  # NOTE: no header


def get_fingerprint(smiles: str) -> np.ndarray:
    mol = AllChem.MolFromSmiles(smiles)
    assert mol is not None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
    return np.array(fp, dtype=np.float32)


class PaRoutesInventory(SmilesListInventory):
    def __init__(self, n: int = 5, **kwargs):
        # Load stock molecules
        with open(STOCK_FILES[n]) as f:
            lines = f.readlines()
        stock_smiles = lines[1:]  # skip header
        super().__init__(smiles_list=stock_smiles, canonicalize=True, **kwargs)


class PaRoutesModel(BackwardReactionModel):
    def __init__(self, max_num_templates: int = 50, **kwargs):
        super().__init__(**kwargs)
        self.max_num_templates = max_num_templates
        self._build_model()

    def _build_model(
        self,
    ):
        """Builds template classification model."""

        # Template library
        self._template_df = pd.read_csv(UNIQUE_TEMPLATE_FILE, sep="\t", compression="gzip")

        # Keras model
        self._model = keras.models.load_model(
            MODEL_DUMP_FILE,
            custom_objects={
                "top10_acc": keras.metrics.TopKCategoricalAccuracy(k=10, name="top10_acc"),
                "top50_acc": keras.metrics.TopKCategoricalAccuracy(k=50, name="top50_acc"),
            },
        )

    def _get_backward_reactions(self, mols: list[Molecule]) -> list[list[BackwardReaction]]:
        # Make fingerprint array
        fingperprints = np.array([get_fingerprint(mol.smiles) for mol in mols])

        # Call model
        template_softmax = self._model(fingperprints, training=False).numpy()
        assert template_softmax.shape == (len(mols), len(self._template_df))
        template_argsort = np.argsort(-template_softmax, axis=1)

        # Run reactions for most likely templates
        output: list[list[BackwardReaction]] = []
        for mol_i, mol in enumerate(mols):
            curr_rxn_list: list[BackwardReaction] = []
            for template_rank in range(self.max_num_templates):
                # Get template at this rank
                template_idx = template_argsort[mol_i, template_rank]
                curr_template_row = self._template_df.iloc[template_idx]
                curr_softmax_value = float(template_softmax[mol_i, template_idx])

                # Run reaction
                reactants_list: list[str] = rdchiralRunText(
                    curr_template_row.retro_template, mol.smiles
                )

                # Filter out possible duplicates (the same template can be used multiple times but give the same reactants)
                reactant_sets = set(frozenset(s.split(".")) for s in reactants_list)

                # Create reaction outputs
                for reactant_strs in reactant_sets:
                    reactant_mols = [
                        Molecule(smiles=s, make_rdkit_mol=False) for s in reactant_strs
                    ]
                    curr_rxn_list.append(
                        BackwardReaction(
                            reactants=frozenset(reactant_mols),
                            product=mol,
                            metadata={
                                "template": curr_template_row.retro_template,
                                "softmax": curr_softmax_value,  # type: ignore[typeddict-item]
                                "template_idx": template_idx,  # type: ignore[typeddict-item]
                                "template_rank": template_rank,  # type: ignore[typeddict-item]
                                "template_library_occurence": curr_template_row.library_occurence,  # type: ignore[typeddict-item]
                            },
                        )
                    )
            output.append(curr_rxn_list)

        return output
