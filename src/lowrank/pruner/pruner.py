import enum

from tensorflow.keras import models
from lowrank.low_rank_layer import LowRankLayer


class PruningScope(enum.Enum):
    GLOBAL = enum.auto()  # global pruning will score all ranks from all layers together
    LOCAL = enum.auto()  # local pruning will treat each layer independently


class Pruner:
    """
    Pruners take a model, and upon examining its effective weights, computes rank masks for
    each layer
    """
    def __init__(self, model: models.Sequential, scope: PruningScope, sparsity: float):
        self.model = model
        self.scope = scope
        if sparsity < 0 or sparsity > 1:
            raise ValueError("Sparsity must be in the range [0, 1]")
        self.sparsity = sparsity

    def compute_masks(self) -> list[list[bool]]:
        """
        Computes and returns a list of masks for each layer in the model.
        """
        raise NotImplementedError("Must be called on a subclass of Pruner")

    def prune(self):
        """
        Calls the `compute_mask` method and actually sets the ranks
        """
        masks = self.compute_masks()
        low_rank_layers: list[LowRankLayer] = list(
            filter(lambda x: isinstance(x, LowRankLayer), self.model.layers)
        )
        if len(masks) != len(low_rank_layers):
            raise ValueError("Computed mask does not match length of model layers")
        for mask, layer in zip(masks, low_rank_layers):
            layer.set_rank(mask)
