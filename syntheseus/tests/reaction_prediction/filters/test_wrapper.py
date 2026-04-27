"""Tests for `FilteredBackwardReactionModel` using dummy backward and filter models."""
from __future__ import annotations

from syntheseus.interface.molecule import Molecule
from syntheseus.reaction_prediction.filters.wrapper import FilteredBackwardReactionModel
from syntheseus.reaction_prediction.inference.toy_models import LinearMoleculesToyModel
from syntheseus.tests.interface.test_filter_models import ReactantListFilterModel


def test_filtered_backward_model_filters_reactions() -> None:
    """Filter is applied correctly."""
    backward_model = LinearMoleculesToyModel()
    wrapped = FilteredBackwardReactionModel(
        backward_model=backward_model,
        filter_models={"only_C": ReactantListFilterModel(["C"])},
    )

    # Model returns 3 reactions with reactants `[C]`, `[CO]`, `[CS]`; only the first is accepted.
    [filtered] = wrapped([Molecule("CC")])
    assert len(filtered) == 1
    assert [r.smiles for r in filtered[0].reactants] == ["C"]


def test_filtered_backward_model_acceptance_rates() -> None:
    """Acceptance rates (overall and per filter) reflect what the filters reject."""
    backward_model = LinearMoleculesToyModel()

    filter_a = ReactantListFilterModel(["C", "CC", "CO", "CS"])
    filter_b = ReactantListFilterModel(["C"])
    wrapped = FilteredBackwardReactionModel(
        backward_model=backward_model,
        filter_models={"a": filter_a, "b": filter_b},
    )

    # Model returns 3 reactions with reactants `[C, CC]`, `[CCO]`, `[CCS]`; `filter_a` accepts only
    # the first one (1/3), then `filter_b` rejects it (0/1).
    [output] = wrapped([Molecule("CCC")])
    assert output == []
    assert wrapped.acceptance_rate == 0.0
    assert wrapped.acceptance_rate_per_filter == {"a": 1 / 3, "b": 0.0}


def test_filtered_backward_model_chains_filters_in_order() -> None:
    """Once a filter rejects everything, subsequent filters are not called."""
    backward_model = LinearMoleculesToyModel(allow_substitution=False)

    reject_all = ReactantListFilterModel([])
    never_called = ReactantListFilterModel(["C"])
    wrapped = FilteredBackwardReactionModel(
        backward_model=backward_model,
        filter_models={"reject": reject_all, "never": never_called},
    )

    [output] = wrapped([Molecule("CC")])
    assert output == []
    assert reject_all.num_calls() == 1
    assert never_called.num_calls() == 0
    assert wrapped.acceptance_rate_per_filter == {"reject": 0.0, "never": 0.0}


def test_filtered_backward_model_reset_propagates() -> None:
    """Resetting clears all caches and counters."""
    backward_model = LinearMoleculesToyModel(use_cache=True)
    filter = ReactantListFilterModel(["C"], use_cache=True)
    wrapped = FilteredBackwardReactionModel(
        backward_model=backward_model,
        filter_models={"filter": filter},
    )

    wrapped([Molecule("CC")])

    # Backward model called once on `CC`; filter called once per unique reaction (3).
    assert backward_model.num_calls() == 1
    assert filter.num_calls() == 3
    assert wrapped._num_seen == 3
    assert wrapped._num_accepted == 1

    wrapped.reset()
    assert wrapped.num_calls() == 0
    assert backward_model.num_calls() == 0
    assert filter.num_calls() == 0
    assert wrapped._num_seen == 0
    assert wrapped._num_accepted == 0
    assert wrapped.acceptance_rate == 0.0
    assert wrapped.acceptance_rate_per_filter == {"filter": 0.0}


def test_filtered_backward_model_no_filters_passthrough() -> None:
    """With no filters, output matches the underlying backward model."""
    backward_model = LinearMoleculesToyModel(allow_substitution=False)
    wrapped = FilteredBackwardReactionModel(backward_model=backward_model, filter_models={})

    inputs = [Molecule("CC"), Molecule("CCC")]
    assert wrapped(inputs) == backward_model(inputs)
