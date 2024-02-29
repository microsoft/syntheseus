from __future__ import annotations

import copy
from dataclasses import FrozenInstanceError

import pytest

from syntheseus.interface.bag import Bag
from syntheseus.interface.molecule import Molecule
from syntheseus.interface.reaction import Reaction, SingleProductReaction


def test_reaction_objects_basic():
    """Instantiate a `Reaction` object."""
    C2 = Molecule(2 * "C")
    C3 = Molecule(3 * "C")
    C5 = Molecule(5 * "C")

    # Single product reaction
    rxn1 = SingleProductReaction(
        reactants=Bag([C2, C3]),
        product=C5,
    )
    assert rxn1.reaction_smiles == "CC.CCC>>CCCCC"

    # Standadr (multi-product) reaction
    rxn2 = Reaction(
        reactants=Bag([C5]),
        products=Bag([C2, C3]),
    )
    assert rxn2.reaction_smiles == "CCCCC>>CC.CCC"


class TestSingleProductReactions:
    def test_positive_equality(self, rxn_cocs_from_co_cs: SingleProductReaction) -> None:
        """Various tests that 2 reactions with same products and reactants should be equal."""

        rxn_copy = SingleProductReaction(
            product=copy.deepcopy(rxn_cocs_from_co_cs.product),
            reactants=copy.deepcopy(rxn_cocs_from_co_cs.reactants),
        )

        # Test 1: original and copy should be equal
        assert rxn_copy == rxn_cocs_from_co_cs

        # Test 2: although equal, they should be distinct objects
        assert rxn_cocs_from_co_cs is not rxn_copy

        # Test 3: differences in metadata should not affect equality
        rxn_cocs_from_co_cs.metadata["test"] = "str1"  # type: ignore[typeddict-unknown-key]
        rxn_copy.metadata["test"] = "str2"  # type: ignore[typeddict-unknown-key]
        assert rxn_copy == rxn_cocs_from_co_cs
        assert rxn_cocs_from_co_cs.metadata != rxn_copy.metadata

    def test_negative_equality(self, rxn_cocs_from_co_cs: SingleProductReaction) -> None:
        """Various tests that reactions which should not be equal are not equal."""

        # Test 1: changing identifier makes reactions not equal
        rxn_with_different_id = SingleProductReaction(
            product=copy.deepcopy(rxn_cocs_from_co_cs.product),
            reactants=copy.deepcopy(rxn_cocs_from_co_cs.reactants),
            identifier="different",
        )
        assert rxn_cocs_from_co_cs != rxn_with_different_id

        # Test 2: different products and reactions are not equal
        diff_rxn = SingleProductReaction(
            product=Molecule("CC"), reactants=Bag([Molecule("CO"), Molecule("CS")])
        )
        assert rxn_cocs_from_co_cs != diff_rxn

    def test_frozen(self, rxn_cocs_from_co_cs: SingleProductReaction) -> None:
        """Test that the fields of the reaction are frozen."""
        with pytest.raises(FrozenInstanceError):
            # type ignore is because mypy complains we are modifying a frozen field, which is the point of the test
            rxn_cocs_from_co_cs.identifier = "abc"  # type: ignore[misc]

    def test_rxn_smiles(self, rxn_cocs_from_co_cs: SingleProductReaction) -> None:
        """Test that the reaction SMILES is as expected."""
        assert rxn_cocs_from_co_cs.reaction_smiles == "CO.CS>>COCS"
