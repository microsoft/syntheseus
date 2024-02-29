from __future__ import annotations

import warnings
from abc import abstractmethod
from typing import Any, Generic, Optional, Sequence, TypeVar

from syntheseus.interface.bag import Bag
from syntheseus.interface.molecule import Molecule
from syntheseus.interface.reaction import Reaction, SingleProductReaction

InputType = TypeVar("InputType")
ReactionType = TypeVar("ReactionType", bound=Reaction)


DEFAULT_NUM_RESULTS = 100


class ReactionModel(Generic[InputType, ReactionType]):
    """Base class for all reaction models, both backward and forward."""

    def __init__(
        self,
        *,
        remove_duplicates: bool = True,
        use_cache: bool = False,
        count_cache_in_num_calls: bool = False,
        initial_cache: Optional[dict[tuple[InputType, int], Sequence[ReactionType]]] = None,
        default_num_results: int = DEFAULT_NUM_RESULTS,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)  # in case this is not the first class in the MRO
        self.count_cache_in_num_calls = count_cache_in_num_calls
        self.default_num_results = default_num_results

        # Attributes used in caching. They should not be modified manually
        # since doing so will likely make counts/etc inaccurate
        self._use_cache = False  # dummy init, will be set in reset
        self._cache: dict[tuple[InputType, int], Sequence[ReactionType]] = dict()
        self._remove_duplicates = remove_duplicates
        self.reset(use_cache=use_cache)

        # Add initial cache *after* reset is done so it is not cleared
        if self._use_cache:
            self._cache.update(
                initial_cache or dict()
            )  # syntactic sugar to avoid if ... is None check
        elif initial_cache is not None:
            warnings.warn(
                "An initial cache was provided but will be ignored because caching is turned off.",
                category=UserWarning,
            )

    def reset(self, use_cache: Optional[bool] = None) -> None:
        """Reset counts, caches, etc for this model."""
        self._cache.clear()

        # Potential reset of use_cache
        if use_cache is not None:
            self._use_cache = use_cache

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

    def __call__(
        self, inputs: list[InputType], num_results: Optional[int] = None
    ) -> list[Sequence[ReactionType]]:
        """Given a batch of inputs to the reaction model, return a batch of results.

        Args:
            inputs: Batch of inputs to the reaction model, each either a molecule or a set of
                molecules, depending on directionality.
            num_results: Number of results to return for each input in the batch. Many models may
                only be able to produce a finite number of candidate outputs, thus the returned
                lists are allowed to be shorter than `num_results`. If not provided, the default
                number of results will be used.
        """

        # Step 0: set num_results to default if not provided
        num_results = num_results or self.default_num_results

        # Step 1: call underlying model for all inputs not in the cache,
        # and add them to the cache
        inputs_not_in_cache = list({inp for inp in inputs if (inp, num_results) not in self._cache})
        if len(inputs_not_in_cache) > 0:
            new_rxns = self._get_reactions(inputs=inputs_not_in_cache, num_results=num_results)
            assert len(new_rxns) == len(inputs_not_in_cache)
            for inp, rxns in zip(inputs_not_in_cache, new_rxns):
                self._cache[(inp, num_results)] = self.filter_reactions(rxns)

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
    def _get_reactions(
        self, inputs: list[InputType], num_results: int
    ) -> list[Sequence[ReactionType]]:
        """
        Method to override which returns the underlying reactions.
        It is encouraged (but not mandatory) that the order of reactions is stable
        to minimize variation between runs.
        """

    def filter_reactions(self, reaction_list: Sequence[ReactionType]) -> Sequence[ReactionType]:
        """
        Filters a list of reactions. In the base version this just removes duplicates,
        but subclasses could add additional behaviour or override this.
        """
        if self._remove_duplicates:
            # This removes duplicates but preserves order since dict's
            # insertion order is preserved in Python 3.7+
            return list(dict.fromkeys(reaction_list))
        else:
            return list(reaction_list)

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


class ForwardReactionModel(ReactionModel[Bag[Molecule], Reaction]):
    def is_forward(self) -> bool:
        return True
