from pathlib import Path
from typing import Any, Dict, List, Sequence, Union

from syntheseus.interface.models import Prediction
from syntheseus.reaction_prediction.chem.utils import molecule_bag_from_smiles


def process_raw_smiles_outputs(
    input: Any, output_list: List[str], kwargs_list: List[Dict[str, Any]]
) -> Sequence[Prediction]:
    """Convert raw SMILES outputs into a list of `Prediction` objects.

    Args:
        inputs: Model input (can be `Molecule` or `Bag[Molecule]` depending on directionality).
        output_list: Raw SMILES outputs (including potentially invalid ones).
        kwargs_list: Additional metadata to attach to the predictions (e.g. probability).

    Returns:
        A list of `Prediction`s; may be shorter than `outputs` if some of the raw
        SMILES could not be parsed into valid reactant bags.
    """
    predictions: List[Prediction] = []

    for raw_output, kwargs in zip(output_list, kwargs_list):
        reactants = molecule_bag_from_smiles(raw_output)

        # Only consider the prediction if the SMILES can be parsed.
        if reactants is not None:
            predictions.append(Prediction(input=input, output=reactants, **kwargs))

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
