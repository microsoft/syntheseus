"""Code related to starting molecules match metric (called exact set-wise match in FusionRetro)."""

from __future__ import annotations

import itertools
import typing
from collections.abc import Iterable

T = typing.TypeVar("T")


def partition_set(A: list[T], k: int) -> Iterable[list[list[T]]]:
    """
    Enumerates all possible ways to partition a list A into k disjoint subsets which are non-empty
    (in all possible orders).

    This function is easiest to explain by example:

    A single partition is just the list A itself.

    >>> list(partition_set([1, 2, 3], 1))
    >>> [[[1, 2, 3]]]

    For k=2, there are 3 possible partitions each with two possible orders.

    >>> list(partition_set([1, 2, 3], 2))
    >>> [[[1], [2, 3]], [[2], [1, 3]], [[3], [1, 2]], [[1, 2], [3]], [[1, 3], [2]], [[2, 3], [1]]]

    For k=3, there is only 1 possible partition but 6 possible orders.

    >>> list(partition_set([1, 2, 3], 3))
    >>> [[[1], [2], [3]], [[1], [3], [2]], [[2], [1], [3]], [[2], [3], [1]], [[3], [1], [2]], [[3], [2], [1]]]

    This function uses a recursive implementation.
    First it partitions A into 2 (where the second partition has at least k-1 elements),
    then it recursively partitions the second partition into (k-1) partitions.
    """

    # Check 1: elements of list are unique
    assert len(set(A)) == len(A)

    # Check 2: k is valid
    assert k >= 1

    # Base case 1: list is empty
    if len(A) == 0:
        return

    # Base case 2: just a single partition
    if k == 1:
        yield [A]
        return

    # Main case: partition A into two parts, then recursively partition the second part
    max_size_of_first_partition = len(A) - k + 1
    for first_partition_size in range(1, max_size_of_first_partition + 1):
        for first_partition in itertools.combinations(A, first_partition_size):
            # Find the remaining elements to partition.
            # NOTE: this assumes that the elements of A are unique.
            remaining_elements = [x for x in A if x not in first_partition]
            for subsequent_partitions in partition_set(remaining_elements, k - 1):
                yield [list(first_partition)] + subsequent_partitions
