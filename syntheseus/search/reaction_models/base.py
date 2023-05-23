from __future__ import annotations

import abc
import warnings
from typing import Optional

from syntheseus.search.chem import BackwardReaction, Molecule


def remove_duplicate_reactions(reaction_list: list[BackwardReaction]) -> list[BackwardReaction]:
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
    seen_reactions: set[BackwardReaction] = set()
    list_out: list[BackwardReaction] = list()
    for rxn in reaction_list:
        if rxn not in seen_reactions:
            seen_reactions.add(rxn)
            list_out.append(rxn)
    return list_out


class BackwardReactionModel(abc.ABC):
    """
    Base class for reaction models. Generally wraps an existing reaction model
    and adds caching, tracking of number of calls, and filtering.
    """

    def __init__(
        self,
        *,
        remove_duplicates: bool = True,
        use_cache: bool = True,
        count_cache_in_num_calls: bool = False,
        initial_cache: Optional[dict[Molecule, list[BackwardReaction]]] = None,
    ) -> None:
        self.count_cache_in_num_calls = count_cache_in_num_calls

        # These attributes should not be modified manually,
        # since doing so will likely make counts/etc inaccurate
        self._use_cache = use_cache
        self._cache: dict[Molecule, list[BackwardReaction]] = dict()
        self._remove_duplicates = remove_duplicates
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

    def filter_reactions(self, reaction_list: list[BackwardReaction]) -> list[BackwardReaction]:
        """
        Filters a list of reactions. In the base version this just removes duplicates,
        but subclasses could add additional behaviour or override this.
        """
        if self._remove_duplicates:
            return remove_duplicate_reactions(reaction_list)
        else:
            return list(reaction_list)

    def __call__(self, mols: list[Molecule]) -> list[list[BackwardReaction]]:
        """Return all backward reactions."""

        # Step 1: call underlying model for all mols not in the cache,
        # and add them to the cache
        mols_not_in_cache = list({m for m in mols if m not in self._cache})
        if len(mols_not_in_cache) > 0:
            new_rxns = self._get_backward_reactions(mols_not_in_cache)
            assert len(new_rxns) == len(mols_not_in_cache)
            for mol, rxns in zip(mols_not_in_cache, new_rxns):
                self._cache[mol] = self.filter_reactions(rxns)

        # Step 2: all reactions should now be in the cache,
        # so the output can just be assembled from there.
        # Clear the cache if use_cache=False
        output = [self._cache[mol] for mol in mols]
        if not self._use_cache:
            self._cache.clear()

        # Step 3: increment counts
        self._num_cache_misses += len(mols_not_in_cache)
        self._num_cache_hits += len(mols) - len(mols_not_in_cache)

        return output

    @abc.abstractmethod
    def _get_backward_reactions(self, mols: list[Molecule]) -> list[list[BackwardReaction]]:
        """
        Method to override which returns the underlying reactions.
        It is encouraged (but not mandatory) that the order of reactions is stable
        to minimize variation between runs.
        """
