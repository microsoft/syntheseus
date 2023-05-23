from syntheseus.search.algorithms.base import MolSetSearchAlgorithm
from syntheseus.search.algorithms.mcts.base import BaseMCTS
from syntheseus.search.graph.molset import MolSetGraph, MolSetNode


class MolSetMCTS(BaseMCTS[MolSetGraph, MolSetNode, MolSetNode], MolSetSearchAlgorithm[int]):
    pass
