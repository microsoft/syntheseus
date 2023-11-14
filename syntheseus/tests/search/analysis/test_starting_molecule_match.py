import pytest

from syntheseus.search.analysis import starting_molecule_match


@pytest.mark.parametrize(
    "A,k,expected_partitions",
    [
        ([1, 2, 3], 1, [[[1, 2, 3]]]),
        (
            [1, 2, 3],
            2,
            [
                [[1], [2, 3]],
                [[2], [1, 3]],
                [[3], [1, 2]],
                [[1, 2], [3]],
                [[1, 3], [2]],
                [[2, 3], [1]],
            ],
        ),
        (
            [1, 2, 3],
            3,
            [
                [[1], [2], [3]],
                [[1], [3], [2]],
                [[2], [1], [3]],
                [[2], [3], [1]],
                [[3], [1], [2]],
                [[3], [2], [1]],
            ],
        ),
        ([1, 2, 3], 4, []),
        ([1, 2, 3], 5, []),
        (
            [1, 2, 3, 4],
            2,
            [
                [[1], [2, 3, 4]],
                [[2], [1, 3, 4]],
                [[3], [1, 2, 4]],
                [[4], [1, 2, 3]],
                [[1, 2], [3, 4]],
                [[1, 3], [2, 4]],
                [[1, 4], [2, 3]],
                [[2, 3], [1, 4]],
                [[2, 4], [1, 3]],
                [[3, 4], [1, 2]],
                [[1, 2, 3], [4]],
                [[1, 2, 4], [3]],
                [[1, 3, 4], [2]],
                [[2, 3, 4], [1]],
            ],
        ),
        ([], 1, []),  # empty list has no partitions
        ([], 2, []),  # test again with k=2
    ],
)
def test_partition_set_valid(A, k, expected_partitions):
    output = list(starting_molecule_match.partition_set(A, k))
    assert output == expected_partitions


@pytest.mark.parametrize(
    "A,k",
    [
        ([1, 2, 3], -1),  # negative k
        ([1, 2, 3], 0),  # k=0
        ([1, 2, 2], 0),  # list has duplicates
    ],
)
def test_partition_set_invalid(A, k):
    with pytest.raises(AssertionError):
        list(starting_molecule_match.partition_set(A, k))
