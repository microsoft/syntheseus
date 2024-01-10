from __future__ import annotations

import math
import warnings
from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Generic, List, Optional, TypeVar

from syntheseus.interface.bag import Bag
from syntheseus.interface.molecule import REACTION_SEPARATOR, SMILES_SEPARATOR, Molecule
from syntheseus.interface.typed_dict import TypedDict

InputType = TypeVar("InputType")
OutputType = TypeVar("OutputType")
R = TypeVar("R")
DEFAULT_NUM_RESULTS = 100


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

    def __init__(
        self,
        *,
        default_num_results: int = DEFAULT_NUM_RESULTS,
        remove_duplicates: bool = True,
        use_cache: bool = True,
        count_cache_in_num_calls: bool = False,
        initial_cache: Optional[dict] = None,
    ) -> None:
        self.count_cache_in_num_calls = count_cache_in_num_calls

        # These attributes should not be modified manually,
        # since doing so will likely make counts/etc inaccurate
        self._use_cache = use_cache
        self._cache: dict[tuple[InputType, int], PredictionList[InputType, OutputType]] = dict()
        self._remove_duplicates = remove_duplicates
        self.default_num_results = default_num_results
        self.reset()

        # Add initial cache after reset is done
        if self._use_cache:
            self._cache.update(
                initial_cache or dict()
            )  # syntactic sugar to avoid if ... is None check
        elif initial_cache is not None:
            warnings.warn(
                "An initial cache was provided but will be ignored because caching is turned off.",
                category=UserWarning,
            )

    def reset(self) -> None:
        """Reset counts, caches, etc for this model."""
        self._cache.clear()

        # Cache counts, using same terminology as LRU cache
        # hit = was in the cache
        # miss = was not in the cache
        self._num_cache_hits = 0
        self._num_cache_misses = 0

    def num_calls(self, count_cache: Optional[bool] = None) -> int:
        """
        Number of times this reaction model has been called.

        Args:
            count_cache: if true, all calls to the reaction model are counted,
                even if those calls just retrieved an item from the cache.
                If false, just count calls which were not in the cache.
                If None, then `self.count_cache_in_num_calls` is used.
                Defaults to None.

        Returns:
            An integer representing the number of calls to the reaction model.
        """
        if count_cache is None:  # fill in default value
            count_cache = self.count_cache_in_num_calls

        if count_cache:
            return self._num_cache_hits + self._num_cache_misses
        else:
            return self._num_cache_misses

    def filter_reactions(
        self, reaction_list: PredictionList[InputType, OutputType]
    ) -> PredictionList[InputType, OutputType]:
        """
        Filters a list of reactions. In the base version this just removes duplicates,
        but subclasses could add additional behaviour or override this.

        NOTE: the input PredictionList is modified in-place.
        """

        if self._remove_duplicates:
            new_predictions = remove_duplicate_reactions(reaction_list.predictions)
        else:
            new_predictions = list(reaction_list.predictions)

        return PredictionList(
            input=reaction_list.input,
            predictions=new_predictions,
            metadata=reaction_list.metadata,
        )

    def __call__(
        self, inputs: List[InputType], num_results: Optional[int] = None
    ) -> List[PredictionList[InputType, OutputType]]:
        """Given a batch of inputs to the reaction model, return a batch of results.

        Args:
            inputs: Batch of inputs to the reaction model, each either a molecule or a set of
                molecules, depending on directionality.
            num_results: Number of results to return for each input in the batch. Many models may
                only be able to produce a finite number of candidate outputs, thus the returned
                lists are allowed to be shorter than `num_results`.
        """
        num_results = num_results or self.default_num_results

        # Step 1: call underlying model for all inputs not in the cache,
        # and add them to the cache
        inputs_not_in_cache = list({m for m in inputs if (m, num_results) not in self._cache})
        if len(inputs_not_in_cache) > 0:
            new_rxns = self._get_prediction_list(inputs_not_in_cache, num_results=num_results)
            assert len(new_rxns) == len(inputs_not_in_cache)
            for mol, rxns in zip(inputs_not_in_cache, new_rxns):
                self._cache[(mol, num_results)] = self.filter_reactions(rxns)

        # Step 2: all reactions should now be in the cache,
        # so the output can just be assembled from there.
        # Clear the cache if use_cache=False
        output = [self._cache[(inp, num_results)] for inp in inputs]
        if not self._use_cache:
            self._cache.clear()

        # Step 3: increment counts
        self._num_cache_misses += len(inputs_not_in_cache)
        self._num_cache_hits += len(inputs) - len(inputs_not_in_cache)

        return output

    @abstractmethod
    def _get_prediction_list(
        self, inputs: list[InputType], num_results: int
    ) -> list[PredictionList[InputType, OutputType]]:
        """
        Method to override which returns the underlying reactions.
        It is encouraged (but not mandatory) that the order of reactions is stable
        to minimize variation between runs.
        """

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


def remove_duplicate_reactions(reaction_list: list[R]) -> list[R]:
    """
    Remove reactions with the same product/reactants, since these are effectively
    redundant. E.g., if the input is something like
    (noting that types are not actually str):

    ["A+B->C,cost=1.0", "D+E->F,cost=10.0", "A+B->C,cost=5.0"]

    Then the return will be the same list but with the second "A+B->C" removed:
    ["A+B->C,cost=1.0", "D+E->F,cost=10.0"]

    Args:
        reaction_list: list of reactions to filter

    Returns:
        A list identical to `reaction_list`, except with the second + further
        copies of every reaction removed.
    """
    seen_reactions: set[R] = set()
    list_out: list[R] = list()
    for rxn in reaction_list:
        if rxn not in seen_reactions:
            seen_reactions.add(rxn)
            list_out.append(rxn)
    return list_out
