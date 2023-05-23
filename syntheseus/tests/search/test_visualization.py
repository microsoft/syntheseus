"""
Tests that vizualization runs. Tests are skipped if graphviz is not installed.
"""

import tempfile

import pytest

from syntheseus.search.graph.and_or import AndOrGraph
from syntheseus.search.graph.molset import MolSetGraph

visualization = pytest.importorskip("syntheseus.search.visualization")


@pytest.mark.parametrize("draw_mols", [False, True])
@pytest.mark.parametrize("partial_graph", [False, True])
def test_andor_visualization(
    andor_graph_non_minimal: AndOrGraph, draw_mols: bool, partial_graph: bool
) -> None:
    if partial_graph:
        nodes = [n for n in andor_graph_non_minimal.nodes() if n.depth <= 2]
    else:
        nodes = None

    # Visualize, both with and without drawing mols
    with tempfile.TemporaryDirectory() as tmp_dir:
        visualization.visualize_andor(
            graph=andor_graph_non_minimal,
            filename=f"{tmp_dir}/tmp.pdf",
            draw_mols=draw_mols,
            nodes=nodes,
        )


@pytest.mark.parametrize("draw_mols", [False, True])
@pytest.mark.parametrize("partial_graph", [False, True])
def test_molset_visualization(
    molset_tree_non_minimal: MolSetGraph, draw_mols: bool, partial_graph: bool
) -> None:
    if partial_graph:
        nodes = [n for n in molset_tree_non_minimal.nodes() if n.depth <= 2]
    else:
        nodes = None
    with tempfile.TemporaryDirectory() as tmp_dir:
        visualization.visualize_molset(
            graph=molset_tree_non_minimal,
            filename=f"{tmp_dir}/tmp.pdf",
            draw_mols=draw_mols,
            nodes=nodes,
        )


def test_filename_ends_with_pdf(
    molset_tree_non_minimal: MolSetGraph,
    andor_graph_non_minimal: AndOrGraph,
) -> None:
    """Test that an error is raised if the file name doesn't end in .pdf"""

    with pytest.raises(ValueError):
        visualization.visualize_andor(
            graph=andor_graph_non_minimal,
            filename="tmp.xyz",
        )
    with pytest.raises(ValueError):
        visualization.visualize_molset(
            graph=molset_tree_non_minimal,
            filename="tmp.xyz",
        )
