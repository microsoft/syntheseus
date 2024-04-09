from pathlib import Path
from typing import Any, List, Sequence, Union

from syntheseus.interface.bag import Bag
from syntheseus.interface.molecule import Molecule
from syntheseus.interface.reaction import (
    Reaction,
    ReactionMetaData,
    SingleProductReaction,
)
from syntheseus.reaction_prediction.chem.utils import molecule_bag_from_smiles


def process_raw_smiles_outputs_backwards(
    input: Molecule, output_list: List[str], metadata_list: List[ReactionMetaData]
) -> Sequence[SingleProductReaction]:
    """Convert raw SMILES outputs into a list of `SingleProductReaction` objects.

    Args:
        input: Model input.
        output_list: Raw SMILES outputs (including potentially invalid ones).
        metadata_list: Additional metadata to attach to the predictions (e.g. probability).

    Returns:
        A list of `SingleProductReaction`s; may be shorter than `outputs` if some of the raw
        SMILES could not be parsed into valid reactant bags.
    """
    predictions: List[SingleProductReaction] = []

    for raw_output, metadata in zip(output_list, metadata_list):
        reactants = molecule_bag_from_smiles(raw_output)

        # Only consider the prediction if the SMILES can be parsed.
        if reactants is not None:
            predictions.append(
                SingleProductReaction(product=input, reactants=reactants, metadata=metadata)
            )

    return predictions


def process_raw_smiles_outputs_forwards(
    input: Bag[Molecule], output_list: List[str], metadata_list: List[ReactionMetaData]
) -> Sequence[Reaction]:
    """Convert raw SMILES outputs into a list of `Reaction` objects.
    Like method `process_raw_smiles_outputs_backwards`, but for forward models.

    Args:
        input: Model input.
        output_list: Raw SMILES outputs (including potentially invalid ones).
        metadata_list: Additional metadata to attach to the predictions (e.g. probability).

    Returns:
        A list of `Reaction`s; may be shorter than `outputs` if some of the raw
        SMILES could not be parsed into valid reactant bags.
    """
    predictions: List[Reaction] = []

    for raw_output, metadata in zip(output_list, metadata_list):
        products = molecule_bag_from_smiles(raw_output)

        # Only consider the prediction if the SMILES can be parsed.
        if products is not None:
            predictions.append(Reaction(products=products, reactants=input, metadata=metadata))

    return predictions


def get_unique_file_in_dir(dir: Union[str, Path], pattern: str) -> Path:
    candidates = list(Path(dir).glob(pattern))
    if len(candidates) != 1:
        raise ValueError(
            f"Expected a unique match for {pattern} in {dir}, found {len(candidates)}: {candidates}"
        )

    return candidates[0]


def get_module_path(module: Any) -> str:
    """Heuristically extract the local path to an imported module."""

    # In some cases, `module.__path__` is already a `List`, while in other cases it may be a
    # `_NamespacePath` object. Either way the conversion below leaves us with `List[str]`.
    path_list: List[str] = list(module.__path__)

    if len(path_list) != 1:
        raise ValueError(f"Cannot extract path to module {module} from {path_list}")

    return path_list[0]
