"""
SNIP Pruner
"""

from lowrank.pruners import AbstractPrunerBase, create_mask


class SnipPruner(AbstractPrunerBase):
    """
    Class for SNIP Pruner
    Implements compute score to score singular vectors using SNIP Method.
    """

    def compute_scores(self) -> "list[list[int | float]]":
        """
        Score = loss if masking out the singular vector
        Intuition = if loss when masking out the singular vector is high,
        then the singular vector must be important.
        """
        if self.data is None or self.loss is None:
            raise ValueError("Snip pruner requires data and loss function.")

        scores = []
        for layer_ind, layer in enumerate(self.layers_to_prune):
            print(f"Pruning layer: {layer_ind}")
            layer_scores = []
            for i in range(layer.rank_capacity):
                self.set_mask_on_layer(
                    layer, create_mask(layer.rank_capacity, [i], inverted=True)
                )
                loss = self.model.evaluate(
                    self.data_x, self.data_y, self.batch_size, verbose=0
                )[0]
                print(f"\rMasking out SV {i:03}\tloss: {loss:.5f}", end="")
                layer_scores.append(loss)
            print()
            # reset mask of layer before moving onto next layer
            self.set_mask_on_layer(
                layer, create_mask(layer.rank_capacity, [], inverted=True)
            )
            scores.append(layer_scores)
        return scores
