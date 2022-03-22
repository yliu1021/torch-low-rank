import pruner
from lowrank import low_rank_layer


class MagPruner(pruner.Pruner):
    """
    Magnitude pruner scores singular vectors based on magnitude of the vector
    """

    def compute_masks(self) -> list[list[bool]]:
        if self.scope == pruner.PruningScope.GLOBAL:
            raise ValueError("Global pruning not yet supported")
        masks = []
        for layer in self.model.layers:
            if not isinstance(layer, low_rank_layer.LowRankLayer):
                # only care about low rank layers
                continue
            num_drop = int(round(self.sparsity * layer.max_rank))
            mask = [True] * layer.max_rank
            mask[-num_drop:] = [False] * num_drop
            masks.append(mask)
        return masks
