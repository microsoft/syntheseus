from __future__ import annotations

import pytest

from syntheseus.search.algorithms.random import (
    AndOr_RandomSearch,
    MolSet_RandomSearch,
)
from syntheseus.tests.search.algorithms.test_base import BaseAlgorithmTest
from syntheseus.tests.search.conftest import RetrosynthesisTask


class BaseRandomSearchTest(BaseAlgorithmTest):
    """
    Base test for random search.

    We skip `test_found_routesX` because random search is very inefficient.
    """

    @pytest.mark.skip(reason="Random search is very inefficient")
    def test_found_routes1(self, retrosynthesis_task1: RetrosynthesisTask) -> None:
        pass

    @pytest.mark.skip(reason="Random search is very inefficient")
    def test_found_routes2(self, retrosynthesis_task2: RetrosynthesisTask) -> None:
        pass

    @pytest.mark.parametrize("limit", [0, 1, 2, 1000])
    def test_limit_iterations(
        self,
        retrosynthesis_task1: RetrosynthesisTask,
        retrosynthesis_task2: RetrosynthesisTask,
        retrosynthesis_task3: RetrosynthesisTask,
        limit: int,
    ) -> None:
        # Here we are just overriding the limits which are tested.
        # Random search is inefficient, so sometimes after 100 iterations not all tasks are solved.
        super().test_limit_iterations(
            retrosynthesis_task1, retrosynthesis_task2, retrosynthesis_task3, limit
        )


class TestAndOrRandomSearch(BaseRandomSearchTest):
    def setup_algorithm(self, **kwargs) -> AndOr_RandomSearch:
        return AndOr_RandomSearch(**kwargs)


class TestMolSetRandomSearch(BaseRandomSearchTest):
    def setup_algorithm(self, **kwargs) -> MolSet_RandomSearch:
        return MolSet_RandomSearch(**kwargs)
