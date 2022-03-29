"""
Magnitude Pruner by weight
"""
from typing import Iterable, Union

import numpy as np

from lowrank.pruners import AbstractPrunerBase


class WeightMagPruner(AbstractPrunerBase):
    """
    Magnitude pruners scores singular vectors based on magnitude of the vector
    """

    def compute_scores(self) -> list[np.ndarray]:
        scores = []
        for layer in self.layers_to_prune:
            scores.append(np.abs(layer.eff_weight().numpy()))
        return scores
