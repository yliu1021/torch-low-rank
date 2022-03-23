"""
SNIP Pruner
"""

from lowrank.pruners import create_mask, AbstractPrunerBase

class SnipPruner(AbstractPrunerBase):
    '''
    Class for SNIP Pruner
    Implements compute score to score singular vectors using SNIP Method.
    '''
    def compute_scores(self) -> 'list[list[int | float]]':
        """
        Score = loss if masking out the singular vector
        Intuition = if loss when masking out the singular vector is high,
        then the singular vector must be important.
        """
        if self.data is None or self.loss is None:
            raise ValueError("Snip pruner requires data and loss function.")

        scores = []
        for layer in self.layers_to_prune:
            layer_scores = []
            for i in range(layer.rank_capacity):
                layer.set_mask(
                    create_mask(layer.rank_capacity, [i], inverted=True)
                    )
                loss = self.model.evaluate(self.x, self.y, self.batch_size)[0]
                layer_scores.append(loss)
            scores.append(layer_scores)
        return scores
