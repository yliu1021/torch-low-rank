"""
Magnitude Pruner
"""
from typing import Iterable, Union

import numpy as np
from numpy import ndarray

from lowrank.pruners import AbstractPrunerBase


class MagPruner(AbstractPrunerBase):
    """
    Magnitude pruners scores singular vectors based on magnitude of the vector
    """

    def compute_scores(self) -> list[Union[ndarray, Iterable, list[int], list[float]]]:
        scores = []
        for layer in self.layers_to_prune:
            _, singular_values, _ = np.linalg.svd(
                layer.eff_weight(), full_matrices=False
            )
            scores.append(singular_values)
        return scores
