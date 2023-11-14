import pytest

from syntheseus.search.analysis import starting_molecule_match
from syntheseus.search.chem import Molecule
from syntheseus.search.graph.and_or import AndOrGraph


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


class TestStartingMoleculeMatch:
    @pytest.mark.parametrize(
        "starting_smiles,expected_ans",
        [
            ("COCS", True),  # Is starting molecule
            ("CO.CS", True),  # One of the routes
            ("CC", True),  # Another route
            ("CS.CC", True),  # Another route
            ("CO.CC", True),  # Can be a route if CO occurs twice and is reacted in one of them
            ("COCC.CC", False),  # both mols are in graph, but not part of same route
            ("", False),  # an empty set should always be False
        ],
    )
    def test_small_andorgraph(
        self, andor_graph_non_minimal: AndOrGraph, starting_smiles: str, expected_ans: bool
    ):
        starting_mols = {Molecule(s) for s in starting_smiles.split(".")}
        assert (
            starting_molecule_match.is_route_with_starting_mols(
                andor_graph_non_minimal, starting_mols
            )
            == expected_ans
        )

    @pytest.mark.parametrize(
        "starting_smiles,expected_ans",
        [
            ("CC.COC", True),  # small route, should be in there
            ("CCCO.O", True),  # another route from docstring
            ("CCCO.C", True),  # this route exists (although C is not purchasable here)
            ("CCCOC", True),  # this is just the root node
            ("", False),  # an empty set should always be False
            ("C.O", True),  # should be possible to decompose into just C,O
            ("CCCO.CC", False),  # too many atoms
        ],
    )
    def test_large_andorgraph(
        self, andor_graph_with_many_routes: AndOrGraph, starting_smiles: str, expected_ans: bool
    ):
        starting_mols = {Molecule(s) for s in starting_smiles.split(".")}
        assert (
            starting_molecule_match.is_route_with_starting_mols(
                andor_graph_with_many_routes, starting_mols
            )
            == expected_ans
        )
