import lowrank.pruners
from lowrank import low_rank_layer


class SnipPruner(lowrank.pruners.Pruner):
    def compute_masks(self) -> list[list[bool]]:
        if self.data is None or self.loss is None:
            raise ValueError("Snip pruner requires data and loss function.")
        masks = []
        for layer in self.model.layers:
            if not isinstance(layer, low_rank_layer.LowRankLayer):
                continue
            # explicitly set rank to max rank to force weights into U @ V form
            layer.set_rank([True] * layer.max_rank)
            U, V = layer.kernels[layer.rank]
        return masks
