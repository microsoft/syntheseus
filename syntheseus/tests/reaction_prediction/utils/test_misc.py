from typing import Iterable

import pytest

from syntheseus.reaction_prediction.utils.misc import parallelize


def square(x: int) -> int:
    return x * x


@pytest.mark.parametrize("use_iterator", [False, True])
@pytest.mark.parametrize("num_processes", [0, 2])
def test_parallelize(use_iterator: bool, num_processes: int) -> None:
    inputs: Iterable[int] = [1, 2, 3, 4]
    expected_outputs = [square(x) for x in inputs]

    if use_iterator:
        inputs = iter(inputs)

    assert list(parallelize(square, inputs, num_processes=num_processes)) == expected_outputs
