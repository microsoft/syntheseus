"""Route objects are tested implicity in other tests, so there are only minimal tests for now."""

from syntheseus.search.chem import Molecule


def test_route_starting_molecules(minimal_synthesis_graph):
    assert minimal_synthesis_graph.get_starting_molecules() == {Molecule("CC")}
