"""
Pruner Base Class Implementation and other useful package wide code
"""
import enum
from typing import Optional
from venv import create
import numpy as np
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
        data: 'Optional[tuple[np.ndarray, np.ndarray]]' = None,
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
        self.layers_to_prune: list[LowRankLayer] = list(
            filter(lambda x: isinstance(x, LowRankLayer), self.model.layers)
        )

    def compute_scores(self) -> 'list[list[int | float]]':
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

        for layer in self.layers_to_prune:
            if layer.rank_capacity is None:
                layer.set_rank_capacity(layer.max_rank)

        masks = self.create_masks()

        if len(masks) != len(self.layers_to_prune):
            raise ValueError("Computed mask does not match length of model layers")
        for mask, layer in zip(masks, self.layers_to_prune):
            assert layer.rank_capacity == len(mask), (
                "Computed mask should be the same length as " "rank capacity"
            )
            layer.set_mask(mask)
            layer.squeeze_rank_capacity()
        self.model._reset_compile_cache()  # ensure model is recompiled

    def create_masks(self):
        """
        Create masks for the pruning method.
        Calls compute scores which is implemented by the subclass overriding the base Pruner class.
        """
        scores = self.compute_scores()
        masks = []

        thresholds = []
        if self.scope == PruningScope.LOCAL:
            for i in range(len(self.layers_to_prune)):
                sorted_layer_scores = sorted(scores[i])
                num_to_drop = int(len(scores[i]) * (1 - self.sparsity))
                thresholds.append(sorted_layer_scores[num_to_drop])
        elif self.scope == PruningScope.GLOBAL:
            flattened_sorted_scores = sorted([score for layer_scores in scores for score in layer_scores])
            num_to_drop = int(len(flattened_sorted_scores) * (1 - self.sparsity))
            thresholds = [flattened_sorted_scores[num_to_drop]] * len(self.layers_to_prune)
        else:
            raise NotImplementedError(str(self.scope) + " is not supported yet.")

        for i in range(len(self.layers_to_prune)):
            indices_to_drop = np.where(np.array(scores[i]) < thresholds[i])[0]
            masks.append(create_mask(len(scores[i]), indices_to_drop, inverted=True))

        return masks


def create_mask(length: int, indices: 'list[int]', inverted: bool = False, ):
    """
    Helper function that creates mask given
    """
    mask = [(x in indices) for x in range(length)]
    if inverted:
        mask = [not x for x in mask]
    return mask
