"""
pseudo-code for snip
layer1 -> matrix that transforms 10 -> 20
max_rank := min(10, 20) = 10

layer1.set_rank_capacity(layer1.max_rank)

min_i = ____
for i in range(layer1.rank_capacity or layer1.max_rank):
    layer1.set_mask(1 - one_hot(i))
    score = loss(model)
    min_i = ____
masks.append(one_hot(min_i))
return masks
"""
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
