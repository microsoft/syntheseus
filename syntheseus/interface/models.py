from __future__ import annotations

import warnings
from abc import abstractmethod
from collections import OrderedDict
from typing import Any, Callable, Generic, Optional, Sequence, TypeVar

from syntheseus.interface.bag import Bag
from syntheseus.interface.molecule import Molecule
from syntheseus.interface.reaction import Reaction, SingleProductReaction

InputType = TypeVar("InputType")
ValueType = TypeVar("ValueType")
ReactionType = TypeVar("ReactionType", bound=Reaction)


DEFAULT_NUM_RESULTS = 100


def deduplicate_keeping_order(seq: Sequence) -> list:
    """Deduplicate a sequence while preserving order."""
    return list(dict.fromkeys(seq))  # Dict insertion order is preserved in Python 3.7+


class BaseModel(Generic[InputType, ValueType]):
    """Generic base providing an LRU-style cache and call counting for batched models.

    Subclasses define their own `__call__` (with whatever signature they need) and delegate
    to `_cached_call`, passing in the cache keys for the batch and a `compute` callable that
    produces values for the keys that turned out to be cache misses.
    """

    def __init__(
        self,
        *,
        use_cache: bool = False,
        count_cache_in_num_calls: bool = False,
        initial_cache: Optional[dict[InputType, ValueType]] = None,
        max_cache_size: Optional[int] = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)  # in case this is not the first class in the MRO
        self.count_cache_in_num_calls = count_cache_in_num_calls

        # Attributes used in caching. They should not be modified manually
        # since doing so will likely make counts/etc inaccurate
        self._use_cache = False  # dummy init, will be set in reset
        self._cache: OrderedDict[InputType, ValueType] = OrderedDict()
        self._max_cache_size = max_cache_size
        self.reset(use_cache=use_cache)

        # Add initial cache *after* reset is done so it is not cleared
        if initial_cache is not None:
            if self._use_cache:
                if self._max_cache_size is not None and len(initial_cache) > self._max_cache_size:
                    raise ValueError("Initial cache size exceeds `max_cache_size`.")

                self._cache.update(initial_cache)
            else:
                warnings.warn(
                    "Initial cache was provided but will be ignored because caching is turned off.",
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
        """Number of times this model has been called.

        Args:
            count_cache: if `True`, all calls to the model are counted, even if those calls just
                retrieved an item from the cache. If `False`, just count calls which were not in the
                cache. If `None`, then `self.count_cache_in_num_calls` is used.

        Returns:
            An integer representing the number of calls to the model.
        """
        if count_cache is None:  # fill in default value
            count_cache = self.count_cache_in_num_calls

        if count_cache:
            return self._num_cache_hits + self._num_cache_misses
        else:
            return self._num_cache_misses

    @property
    def cache_size(self) -> int:
        """Return the current size of the cache."""
        return len(self._cache) if self._use_cache else 0

    def _cached_call(
        self,
        inputs: Sequence[InputType],
        compute: Callable[[list[InputType]], Sequence[ValueType]],
    ) -> list[ValueType]:
        """Resolve a batch of cache keys, calling `compute` only for the misses.

        Args:
            inputs: Batch of inputs to the model.
            compute: Called with the deduplicated list of missing keys, must return values in the
                same order. Not called at all if every key is already cached.
        """
        # Step 1: call underlying model for all inputs not in the cache, and add them to the cache.
        inputs_not_in_cache = deduplicate_keeping_order(
            [inp for inp in inputs if inp not in self._cache]
        )

        if len(inputs_not_in_cache) > 0:
            new_values = compute(inputs_not_in_cache)
            assert len(new_values) == len(inputs_not_in_cache)
            for inp, value in zip(inputs_not_in_cache, new_values):
                self._cache[inp] = value

        # Step 2: all values should now be in the cache, so output can be assembled from there.
        output = []
        for inp in inputs:
            output.append(self._cache[inp])
            self._cache.move_to_end(inp)  # mark as most recently used

        # Step 2.1: clear the cache if not used, trim if max size is set.
        if not self._use_cache:
            self._cache.clear()
        elif self._max_cache_size is not None:
            # If the cache is larger than the maximum size, remove the oldest entries.
            while len(self._cache) > self._max_cache_size:
                self._cache.popitem(last=False)

        # Step 3: increment counts.
        self._num_cache_misses += len(inputs_not_in_cache)
        self._num_cache_hits += len(inputs) - len(inputs_not_in_cache)

        return output


class ReactionModel(
    BaseModel[tuple[InputType, int], Sequence[ReactionType]], Generic[InputType, ReactionType]
):
    """Base class for all reaction models, both backward and forward."""

    def __init__(
        self,
        *,
        remove_duplicates: bool = True,
        default_num_results: int = DEFAULT_NUM_RESULTS,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.default_num_results = default_num_results
        self._remove_duplicates = remove_duplicates

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

        # Set `num_results` to default if not provided.
        if num_results is None:
            resolved_num_results = self.default_num_results
        else:
            resolved_num_results = num_results

        # Build cache keys (one per input) and delegate to the cached-call helper.
        keys = [(inp, resolved_num_results) for inp in inputs]

        def compute(missing: list[tuple[InputType, int]]) -> list[Sequence[ReactionType]]:
            new_rxns = self._get_reactions(
                inputs=[k[0] for k in missing], num_results=resolved_num_results
            )
            return [self.filter_reactions(rxns) for rxns in new_rxns]

        return self._cached_call(keys, compute=compute)

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
            return deduplicate_keeping_order(reaction_list)
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


class ReactionScoringModel(BaseModel[Reaction, float]):
    """Base class for models that assign a scalar score to reactions.

    Subclasses implement `_get_scores`, which takes a batch of reactions and returns one float
    per reaction.
    """

    def __call__(self, reactions: Sequence[Reaction]) -> list[float]:
        """Return one score per input reaction."""
        return self._cached_call(reactions, compute=self._get_scores)

    @abstractmethod
    def _get_scores(self, reactions: Sequence[Reaction]) -> list[float]:
        """Compute scores for a batch of deduplicated, uncached reactions."""


class ReactionFilterModel(BaseModel[SingleProductReaction, bool]):
    """Base class for models that filter reactions (e.g. for removing hallucinations).

    Subclasses implement `_get_acceptance`, which takes a batch of reactions and returns a single
    boolean per reaction (`True` = accepted, `False` = rejected).
    """

    def __call__(self, reactions: Sequence[SingleProductReaction]) -> list[bool]:
        """Return a boolean acceptance mask: `True` = accepted, `False` = rejected."""
        return self._cached_call(reactions, compute=self._get_acceptance)

    @abstractmethod
    def _get_acceptance(self, reactions: Sequence[SingleProductReaction]) -> list[bool]:
        """Compute acceptance for a batch of deduplicated, uncached reactions."""
