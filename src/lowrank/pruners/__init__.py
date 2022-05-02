"""
Pruner Base Class Implementation and other useful package wide code
"""
import enum
from typing import List, Optional, Tuple, Union

import numpy as np
import tensorflow as tf
from tensorflow.keras import losses, models

from lowrank.low_rank_layer import LowRankLayer


class PruningScope(enum.Enum):
    """
    Pruning Scope determines how to use scores to rank singular vectors and generate mask.
    Global ranks globally, Local ranks locally
    """

    GLOBAL = enum.auto()  # global pruning will score all ranks from all layers together
    LOCAL = enum.auto()  # local pruning will treat each layer independently


class AbstractPrunerBase:
    """
    Pruners take a model, and upon examining its effective weights, computes rank masks for
    each layer
    """

    def __init__(
        self,
        model: models.Sequential,
        scope: PruningScope,
        sparsity: float,
        data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        batch_size: int = 64,
        loss: Optional[losses.Loss] = None,
    ):
        self.model = model
        self.scope = scope
        if sparsity < 0 or sparsity > 1:
            raise ValueError("Sparsity must be in the range [0, 1]")
        self.sparsity = sparsity
        self.data = data
        if self.data is not None:
            self.data_x, self.data_y = data
        self.batch_size = batch_size
        self.loss = loss
        self.layers_to_prune: List[LowRankLayer] = list(
            filter(lambda x: isinstance(x, LowRankLayer), self.model.layers)
        )

    def compute_scores(self) -> List[np.ndarray]:
        """
        Computes and returns scores for the singular vectors in each layer.
        - High Score = Important Singular Vector
        - Low Score = Unimportant Singular Vector
        """
        raise NotImplementedError("Must be called on a subclass of Pruner")

    def prune(self):
        """
        Calls the `compute_mask` method and actually sets the ranks
        """
        masks = self._compute_masks()
        if len(masks) != len(self.layers_to_prune):
            raise ValueError("Computed mask does not match length of model layers")
        for mask, layer in zip(masks, self.layers_to_prune):
            layer.mask = mask
        self.model._reset_compile_cache()  # ensure model is recompiled

    def _compute_masks(self):
        """
        Create masks for the pruning method.
        Calls compute scores which is implemented by the subclass overriding the base Pruner class.
        Creates mask to drop lowest scores in accordance with sparsity ratio.
        """
        # list of ndarrays, each corresponding to each layer
        scores = self.compute_scores()
        assert len(scores) == len(
            self.layers_to_prune
        ), "Number of scores should equal number of layers we're trying to prune"
        if self.scope == PruningScope.LOCAL:
            thresholds = []
            for i in range(len(self.layers_to_prune)):
                sorted_layer_scores = sorted(scores[i].flatten())
                num_to_drop = int(len(sorted_layer_scores) * self.sparsity)
                thresholds.append(sorted_layer_scores[num_to_drop])
        elif self.scope == PruningScope.GLOBAL:
            flattened_sorted_scores = sorted(
                np.concatenate([score.flatten() for score in scores])
            )
            num_to_drop = int(len(flattened_sorted_scores) * self.sparsity)
            thresholds = [flattened_sorted_scores[num_to_drop]] * len(
                self.layers_to_prune
            )
        else:
            raise NotImplementedError(f"{self.scope} is not supported yet.")
        masks = [score >= threshold for score, threshold in zip(scores, thresholds)]
        return masks

    def _set_rank_capacity_on_layer(
        self, layer: LowRankLayer, capacity: Optional[int] = None
    ):
        layer.set_rank_capacity(capacity)
        self.model._reset_compile_cache()


def create_mask(
    length: int,
    indices: List[int],
    inverted: bool = False,
):
    """
    Helper function that creates mask given
    :param length: Length of bool vector
    :param indices: Indices to set to true (if inverted=False i.e. default) and rest set to false
    :param inverted: set to false (default) default behavior, set to true - element-wise not
    :returns: bool vector with only variables at indices set to true if inverted=False (default)
    """
    mask = [float(x in indices) for x in range(length)]
    if inverted:
        mask = [(1 - x) for x in mask]
    return np.array(mask)
