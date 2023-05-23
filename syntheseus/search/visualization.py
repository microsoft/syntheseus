from __future__ import annotations

import pprint
import tempfile
from collections.abc import Collection
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from graphviz import Digraph
from rdkit import Chem
from rdkit.Chem import Draw

from syntheseus.search.graph.and_or import ANDOR_NODE, AndNode, AndOrGraph, OrNode
from syntheseus.search.graph.molset import MolSetGraph, MolSetNode


@dataclass
class GraphVizNode:
    id: str
    label: str
    border_color: str
    fontsize: str = "10.0"
    shape: str = "box"
    style: str = "filled"
    fillcolor: str = "white"


# Constants/etc
PURCHASABLE = "green"
NOT_PURCHASABLE = "red"
PARTIALLY_PURCHASABLE = "yellow"
NO_PURCHASABLE_INFO = "black"


def _mol_to_image(mol: Chem.Mol) -> tempfile._TemporaryFileWrapper[bytes]:
    img = Draw.MolToImage(mol)
    temp_file = tempfile.NamedTemporaryFile(suffix=".png")
    img.save(temp_file)
    return temp_file


def _write_graphviz_graph(
    nodes: list[GraphVizNode],
    edges: list[tuple[str, str]],
    filename: str,
) -> None:
    # Init visualization graph
    if not filename.endswith(".pdf"):
        raise ValueError("Filename must end in .pdf")
    dotfile_name = filename[:-4]
    G = Digraph("G", filename=dotfile_name)  # strip .pdf
    G.format = "pdf"

    # Draw nodes and edges
    for node in nodes:
        G.node(
            node.id,
            label=node.label,
            fontsize=node.fontsize,
            shape=node.shape,
            style=node.style,
            fillcolor=node.fillcolor,
            color=node.border_color,
        )
    for edge in edges:
        G.edge(edge[0], edge[1], label="")

    # Make pdf
    G.render()

    # Remove intermediate dot file
    dotfile = Path(dotfile_name)
    if dotfile.exists():
        dotfile.unlink()


def visualize_andor(
    graph: AndOrGraph,
    filename: str,
    nodes: Optional[Collection[ANDOR_NODE]] = None,
    draw_mols: bool = True,
) -> None:
    """Visualize an AND/OR graph.

    Args:
        graph: The graph to visualize.
        filename: The filename to save the visualization to. Must end in ".pdf"
        nodes: The nodes to include in the visualization. If None, all nodes are included.
        draw_mols: Whether to draw the molecules in the graph.
    """

    # Extract subgraph
    if nodes is None:
        subgraph = graph._graph
    else:
        subgraph = graph._graph.subgraph(nodes)
    del graph

    # Visualize all nodes
    temp_files = []
    graphviz_nodes = []
    for node in subgraph.nodes:
        rows: list[str] = []
        if isinstance(node, OrNode):
            border_color = PURCHASABLE if node.mol.metadata["is_purchasable"] else NOT_PURCHASABLE
            shape = "ellipse"
            if draw_mols and node.mol.smiles != "":
                img_file = _mol_to_image(Chem.MolFromSmiles(node.mol.smiles))
                temp_files.append(img_file)
                rows.append(f'<TD><IMG SRC="{img_file.name}" SCALE="TRUE"/></TD>')
                del img_file
            else:
                rows.append(f"<TD>SMILES: {node.mol.smiles}</TD>")
        elif isinstance(node, AndNode):
            rows.append("<TD>Reaction</TD>")
            border_color = NO_PURCHASABLE_INFO
            shape = "box"
        else:
            raise TypeError(f"Unknown node type {type(node)}")

        # Metadata
        node_data_str = pprint.pformat(node.data).replace("\n", "<BR/>")
        rows.append(f"<TD>{node_data_str}</TD>")

        # Make label
        label = '<<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0">'
        assert len(rows) > 0
        for row in rows:
            label += f"<TR>{row}</TR>"
        label += "</TABLE>>"
        graphviz_nodes.append(
            GraphVizNode(
                id=str(id(node)),
                label=label,
                border_color=border_color,
                shape=shape,
            )
        )

    # Add edges
    edge_list = []
    for node in subgraph.nodes:
        for child in subgraph.successors(node):
            edge_list.append((str(id(node)), str(id(child))))

    # Make the graph
    _write_graphviz_graph(graphviz_nodes, edge_list, filename)


def visualize_molset(
    graph: MolSetGraph,
    filename: str,
    nodes: Optional[Collection[MolSetNode]] = None,
    draw_mols: bool = True,
) -> None:
    """Visualize a MolSet graph.

    Args:
        graph: The graph to visualize.
        filename: The filename to save the visualization to. Must end in ".pdf"
        nodes: The nodes to include in the visualization. If None, all nodes are included.
        draw_mols: Whether to draw the molecules in the graph.
    """

    # Extract subgraph
    if nodes is None:
        subgraph = graph._graph
    else:
        subgraph = graph._graph.subgraph(nodes)
    del graph

    # Visualize all nodes
    temp_files = []
    graphviz_nodes = []
    for node in subgraph.nodes:
        rows: list[str] = []
        purchasable_set = {mol.metadata["is_purchasable"] for mol in node.mols}
        if all(purchasable_set):
            border_color = PURCHASABLE
        elif any(purchasable_set):
            border_color = PARTIALLY_PURCHASABLE
        else:
            border_color = NOT_PURCHASABLE

        for mol in node.mols:
            if draw_mols and mol.smiles != "":
                img_file = _mol_to_image(Chem.MolFromSmiles(mol.smiles))
                temp_files.append(img_file)
                rows.append(f'<TD><IMG SRC="{img_file.name}" SCALE="TRUE"/></TD>')
                del img_file
            else:
                rows.append(f"<TD>SMILES: {mol.smiles}</TD>")

        # Metadata
        node_data_str = pprint.pformat(node.data).replace("\n", "<BR/>")
        rows.append(f"<TD>{node_data_str}</TD>")

        # Make label
        label = '<<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0">'
        assert len(rows) > 0
        for row in rows:
            label += f"<TR>{row}</TR>"
        label += "</TABLE>>"
        graphviz_nodes.append(
            GraphVizNode(
                id=str(id(node)),
                label=label,
                border_color=border_color,
                shape="ellipse",
            )
        )

    # Add edges
    edge_list = []
    for node in subgraph.nodes:
        for child in subgraph.successors(node):
            edge_list.append((str(id(node)), str(id(child))))

    # Make the graph
    _write_graphviz_graph(graphviz_nodes, edge_list, filename)
