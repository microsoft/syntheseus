from __future__ import annotations

import math
from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Generic, List, Optional, TypeVar

from syntheseus.interface.bag import Bag
from syntheseus.interface.molecule import REACTION_SEPARATOR, SMILES_SEPARATOR, Molecule
from syntheseus.interface.typed_dict import TypedDict

InputType = TypeVar("InputType")
OutputType = TypeVar("OutputType")


class ReactionMetaData(TypedDict, total=False):
    """Class to add typing to optional meta-data fields for reactions."""

    cost: float
    template: str
    source: str  # any explanation of the source of this reaction
    probability: float  # probability for this reaction, used for multi-step search
    score: float  # any kind of score for this reaction (e.g. softmax value, probability)
    confidence: float  # confidence (probability) that this reaction is possible
    template_id: int  # index of the template used to generate this reaction


@dataclass(frozen=True, order=False)
class Prediction(Generic[InputType, OutputType]):
    """Reaction prediction from a model, either a forward or a backward one."""

    # The molecule that the prediction is for and the predicted output:
    input: InputType = field(hash=True, compare=True)
    output: OutputType = field(hash=True, compare=True)
    identifier: Optional[str] = field(default=None, hash=True, compare=True)

    # Optional information that may be useful downstream:
    probability: Optional[float] = field(
        default=None, hash=False, compare=False
    )  # Prior probability.
    log_prob: Optional[float] = field(
        default=None, hash=False, compare=False
    )  # As above, but in log space.
    score: Optional[float] = field(default=None, hash=False, compare=False)  # Any other score.
    reaction: Optional[str] = field(default=None, hash=False, compare=False)  # Reaction smiles.
    rxnid: Optional[int] = field(
        default=None, hash=False, compare=False
    )  # Template id, if applicable.

    # Dictionary to hold additional metadata.
    metadata: ReactionMetaData = field(
        default_factory=lambda: ReactionMetaData(),
        hash=False,
        compare=False,
    )

    def __post_init__(self):
        if self.probability is not None and self.log_prob is not None:
            raise ValueError(
                "Probability can be stored as probability or log probability, not both"
            )

    def get_prob(self) -> float:
        if self.probability is not None:
            return self.probability
        elif self.log_prob is not None:
            return math.exp(self.log_prob)
        else:
            raise ValueError("Prediction does not have associated probability or log prob value.")

    def get_log_prob(self) -> float:
        if self.log_prob is not None:
            return self.log_prob
        elif self.probability is not None:
            return math.log(self.probability)
        else:
            raise ValueError("Prediction does not have associated log prob or probability value.")


@dataclass(frozen=True, order=False)
class PredictionList(Generic[InputType, OutputType]):
    """Several possible predictions."""

    input: InputType
    predictions: List[Prediction[InputType, OutputType]]

    # Dictionary to hold additional metadata.
    metadata: Dict[str, Any] = field(default_factory=dict)

    def truncated(self, num_results: int) -> PredictionList[InputType, OutputType]:
        return PredictionList(
            input=self.input, predictions=self.predictions[:num_results], metadata=self.metadata
        )


class ReactionModel(Generic[InputType, OutputType]):
    """Base class for all reaction models, both backward and forward."""

    @abstractmethod
    def __call__(
        self, inputs: List[InputType], num_results: int
    ) -> List[PredictionList[InputType, OutputType]]:
        """Given a batch of inputs to the reaction model, return a batch of results.

        Args:
            inputs: Batch of inputs to the reaction model, each either a molecule or a set of
                molecules, depending on directionality.
            num_results: Number of results to return for each input in the batch. Many models may
                only be able to produce a finite number of candidate outputs, thus the returned
                lists are allowed to be shorter than `num_results`.
        """
        pass

    def get_model_info(self) -> dict[str, Any]:
        return {}

    @abstractmethod
    def is_forward(self) -> bool:
        pass

    def is_backward(self) -> bool:
        return not self.is_forward()

    def get_parameters(self):
        """Return an iterator over parameters (used for computing total parameter count).

        If accurate reporting of number of parameters during evaluation is not important, subclasses
        are free to e.g. return an empty list.
        """
        raise NotImplementedError()


# Below we define some aliases for forward and backward variants of prediction and model classes.
# Model interfaces use bags of SMILES as output type to allow for salts and disconnected components.


class BackwardReactionModel(ReactionModel[Molecule, Bag[Molecule]]):
    def is_forward(self) -> bool:
        return False


class ForwardReactionModel(ReactionModel[Bag[Molecule], Bag[Molecule]]):
    def is_forward(self) -> bool:
        return True


class BackwardPrediction(Prediction[Molecule, Bag[Molecule]]):
    def __init__(self, *args, **kwargs):
        # Provisionally allow both "product/reactant" and "input/output" to be used.
        # In the future we should standardize on one or the other.
        if "reactants" in kwargs:
            reactants = kwargs.pop("reactants")

            # Provisionally allow reactants to be a set or a bag.
            if isinstance(reactants, (set, frozenset)):
                reactants = Bag(reactants)
            kwargs["output"] = reactants
        if "product" in kwargs:
            kwargs["input"] = kwargs.pop("product")
        super().__init__(*args, **kwargs)

    @property
    def product(self) -> Molecule:
        return self.input

    @property
    def reactants(self) -> Bag[Molecule]:
        return self.output

    @property
    def reactants_combined(self) -> str:
        return SMILES_SEPARATOR.join([reactant.smiles for reactant in sorted(self.reactants)])

    @property
    def reaction_smiles(self) -> str:
        return f"{self.reactants_combined}{2 * REACTION_SEPARATOR}{self.product.smiles}"


ForwardPrediction = Prediction[Bag[Molecule], Bag[Molecule]]

BackwardPredictionList = PredictionList[Molecule, Bag[Molecule]]
ForwardPredictionList = PredictionList[Bag[Molecule], Bag[Molecule]]
