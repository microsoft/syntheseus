"""
Test reaction models.

Note that some testing is done implicitly in the algorithm tests
(i.e. those tests will only pass if the reaction models are working properly).
Therefore the tests here are not super exhaustive: instead they try to explicitly
look for some behaviours which might be overlooked by the algorithm tests.
"""

import pytest

from syntheseus.search.chem import BackwardReaction, Molecule
from syntheseus.search.reaction_models.toy import LinearMolecules, ListOfReactionsModel


@pytest.mark.parametrize("remove", [True, False])
def test_remove_duplicates(remove: bool) -> None:
    """
    Test the 'remove_duplicates' kwarg by calling the LinearMolecules model on "CC".

    With remove_duplicates=True there should (obviously) be no duplicates.

    With remove_duplicates=False there should be some duplicates:
    for example, CC -> CO and CC -> OC are the same reaction written 2 ways.
    """

    # Define mols and reactions
    CC = Molecule("CC")
    rxn_C_C = BackwardReaction(product=CC, reactants=frozenset({Molecule("C")}))
    rxn_CO = BackwardReaction(product=CC, reactants=frozenset({Molecule("CO")}))
    rxn_CS = BackwardReaction(product=CC, reactants=frozenset({Molecule("CS")}))

    # Call reaction model
    rxn_model = LinearMolecules(remove_duplicates=remove)
    output = rxn_model([CC])

    # Check that outputs are what is expected.
    # NOTE: this test depends on the order of the outputs being consistent,
    # which is the case for this toy model but may not be true in general
    if remove:
        assert output == [[rxn_C_C, rxn_CO, rxn_CS]]
    else:
        assert output == [[rxn_C_C, rxn_CO, rxn_CS, rxn_CO, rxn_CS]]


@pytest.mark.parametrize("use_cache", [True, False])
def test_caching(cocs_mol: Molecule, use_cache: bool) -> None:
    """Test all aspects of caching for reaction models."""

    # Call model twice on same molecule
    rxn_model = LinearMolecules(use_cache=use_cache)
    output1 = rxn_model([cocs_mol])
    output2 = rxn_model([cocs_mol])

    # Test 1: regardless of caching, outputs should not change
    assert output1 == output2

    # Test 2: is number of calls correct?
    if use_cache:
        assert rxn_model.num_calls() == 1
    else:
        assert rxn_model.num_calls() == 2

    # Test 3: call the model on a batch of new and old molecules and check that count is correct.
    # This tests that the cache is being used correctly for batches of molecules,
    # including only calling the model on a single copy of each molecule if there are duplicates in the batch.
    CC = Molecule("CC")
    output3 = rxn_model([cocs_mol, CC, cocs_mol, CC])
    assert output3[:1] == output1  # should match initial call
    if use_cache:
        assert rxn_model.num_calls() == 2  # called on 2 unique molecules so far
    else:
        assert (
            rxn_model.num_calls() == 4
        )  # called on 4 total molecules so far (duplicates in batch not counted)

    # Test 4: caching should be based on EQUALITY not IDENTITY of molecules,
    # so a different object for the same molecule should retrieve the same item in the cache
    cocs_mol_copy = Molecule(smiles=cocs_mol.smiles)
    output4 = rxn_model([cocs_mol_copy])
    assert output4 == output1
    if use_cache:
        assert rxn_model.num_calls() == 2  # unchanged from before
    else:
        assert rxn_model.num_calls() == 5

    # Test 5: optional argument to num_calls which also counts cache hits
    assert rxn_model.num_calls(count_cache=True) == 7  # 5 + 2 extra copies in batch from test 4

    # Test 6: does resetting work?
    rxn_model.reset()
    assert rxn_model.num_calls() == 0  # should be 0 no matter what
    assert (
        len(rxn_model._cache) == 0
    )  # cache should actually be empty (NOTE: using private attribute is discouraged)

    # Test 7: calling the model after resetting should have same behaviour as fresh model
    # NOTE: this also tests the behaviour of duplicate molecules in a single batch
    output7 = rxn_model([cocs_mol])
    assert output7 == output1
    assert rxn_model.num_calls() == 1


@pytest.mark.parametrize("use_cache", [True, False])
def test_initial_cache(
    cocs_mol: Molecule, rxn_cocs_from_co_cs: BackwardReaction, use_cache: bool
) -> None:
    """Test that seeding the reaction model with an initial cache works as expected."""

    # Create initial cache.
    # NOTE: this is *not* what the reaction model would actually output if called.
    # If the cache works correctly then it will output this reaction instead of calling the model.
    CC = Molecule("CC")
    initial_cache = {cocs_mol: [rxn_cocs_from_co_cs], CC: []}

    # Create reaction model, potentially checking for warning
    if use_cache:
        rxn_model = LinearMolecules(use_cache=use_cache, initial_cache=initial_cache)
    else:
        # If caching is off then providing an initial cache should raise a warning
        with pytest.warns(UserWarning):
            rxn_model = LinearMolecules(use_cache=use_cache, initial_cache=initial_cache)

    # Call reaction model
    outputs = rxn_model([cocs_mol, CC, Molecule("CCC")])
    if use_cache:
        # outputs should reflect initial cache
        assert outputs[0] == initial_cache[cocs_mol]
        assert outputs[1] == initial_cache[CC]
        assert rxn_model.num_calls() == 1
    else:
        # outputs should ignore initial cache
        assert len(outputs[0]) == 7
        assert len(outputs[1]) == 3
        assert rxn_model.num_calls() == 3
    assert (
        len(outputs[2]) == 3
    )  # not in initial cache, so in both cases model should actually be called

    # In all cases, resetting should clear the cache
    rxn_model.reset()
    assert len(rxn_model._cache) == 0


def test_linear_molecules_invalid_molecule() -> None:
    """
    The LinearMolecules model is only defined on the space of linear molecules.
    If called on a non-linear molecule (e.g. with branching) it is currently set up to throw an error.
    This test ensures that this happens.

    NOTE: in the future the behaviour could be changed to just return an empty list,
    but for a toy example I thought it would be best to alert the user with a warning.
    """
    rxn_model = LinearMolecules()
    with pytest.raises(AssertionError):
        rxn_model([Molecule("CC(C)C")])


def test_list_of_reactions_model(
    rxn_cocs_from_co_cs: BackwardReaction,
    rxn_cocs_from_cocc: BackwardReaction,
    rxn_cs_from_cc: BackwardReaction,
) -> None:
    """Simple test of the ListOfReactionsModel class."""
    model = ListOfReactionsModel([rxn_cocs_from_co_cs, rxn_cocs_from_cocc, rxn_cs_from_cc])
    output = model([Molecule("COCS"), Molecule("CS"), Molecule("CO")])
    assert output == [[rxn_cocs_from_co_cs, rxn_cocs_from_cocc], [rxn_cs_from_cc], []]
