from __future__ import annotations

from abc import abstractmethod
from typing import Any, Generic, List, Sequence, TypeVar

from syntheseus.interface.bag import Bag
from syntheseus.interface.molecule import Molecule
from syntheseus.interface.reaction import MultiProductReaction, Reaction, SingleProductReaction

InputType = TypeVar("InputType")
ReactionType = TypeVar("ReactionType", bound=Reaction)


class ReactionModel(Generic[InputType, ReactionType]):
    """Base class for all reaction models, both backward and forward."""

    @abstractmethod
    def __call__(self, inputs: List[InputType], num_results: int) -> List[Sequence[ReactionType]]:
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


class BackwardReactionModel(ReactionModel[Molecule, SingleProductReaction]):
    def is_forward(self) -> bool:
        return False


class ForwardReactionModel(ReactionModel[Bag[Molecule], MultiProductReaction]):
    def is_forward(self) -> bool:
        return True
