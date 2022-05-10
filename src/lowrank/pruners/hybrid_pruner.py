"""
Hybrid pruner that takes the scoring method of two pruners
"""
from typing import List

import numpy as np
import torch

from lowrank.pruners import AbstractPrunerBase


class HybridPruner(AbstractPrunerBase):
    """
    Hybrid pruner will take the scores from two pruning methods and combine them
    """

    def __init__(pruners: List[AbstractPrunerBase], *args, **kwargs):
        self.pruners = pruners
        super().__init__(*args, **kwargs)

    def compute_scores(self) -> List[np.ndarray]:
        all_scores = [p.compute_scores() for p in self.pruners]
        scores = reduce(lambda a, b: [x*y for x, y in zip(a, b)], all_scores[1:], all_scores[0])
        return scores
