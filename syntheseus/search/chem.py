"""Defines molecules and reactions."""

from __future__ import annotations

from syntheseus.interface.models import BackwardPrediction, ReactionMetaData  # noqa: F401
from syntheseus.interface.molecule import Molecule  # noqa: F401

# For backwards compatibility, we just alias the old names to the new ones.
BackwardReaction = BackwardPrediction
