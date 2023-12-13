import math

import numpy as np

from syntheseus.interface.models import Prediction
from syntheseus.interface.molecule import Molecule

# TODO(kmaziarz): Currently this test mostly checks that importing from `models.py` works, and that
# a `Prediction` object can be instantiated. We should extend it later.


def test_prediction():
    prediction = Prediction(input=Molecule("C"), output=Molecule("CC"), probability=0.5)
    assert np.isclose(prediction.get_prob(), 0.5)
    assert np.isclose(prediction.get_log_prob(), math.log(0.5))
