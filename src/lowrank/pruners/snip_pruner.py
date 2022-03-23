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
    def compute_scores(self) -> 'list[list[int | float]]':
        """
        Score = loss if masking out the singular vector
        Intuition = if loss when masking out the singular vector is high, then the singular vector must be important.
        """
        if self.data is None or self.loss is None:
            raise ValueError("Snip pruner requires data and loss function.")

        scores = []
        for layer in self.layers_to_prune:
            layer_scores = []
            for i in range(layer.rank_capacity):
                layer.set_mask(super.create_mask(layer.rank_capacity, lowrank.pruners.MaskType.INVERTED, [i]))
                loss = self.model.evaluate(self.x, self.y, self.batch_size)[0]
                layer_scores.append(loss)
            scores.append(layer_scores)
        
        return scores



