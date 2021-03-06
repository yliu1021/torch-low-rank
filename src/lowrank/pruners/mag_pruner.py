"""
Magnitude Pruner
"""
from typing import List

import numpy as np

from lowrank.pruners import AbstractPrunerBase


class MagPruner(AbstractPrunerBase):
    """
    Magnitude pruners scores singular vectors based on magnitude of the vector
    """

    def compute_scores(self) -> List[np.ndarray]:
        scores = []
        for layer in self.layers_to_prune:
            _, singular_values, _ = np.linalg.svd(
                layer.eff_weight(), full_matrices=False
            )
            scores.append(singular_values)
        return scores
