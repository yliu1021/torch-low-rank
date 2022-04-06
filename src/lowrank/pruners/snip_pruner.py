"""
SNIP SV Pruner - calculates Delta L instead of gradient estimate
"""
from functools import reduce

import numpy as np
import tensorflow as tf

from lowrank.pruners import AbstractPrunerBase, create_mask


class SnipPruner(AbstractPrunerBase):
    """
    Class for SNIP Pruner
    Implements compute score to score singular vectors using SNIP Method.
    """

    def compute_scores(self) -> "list[np.ndarray]":
        """
        Score = loss if masking out the singular vector
        Intuition = if loss when masking out the singular vector is high,
        then the singular vector must be important.
        """
        if self.data is None or self.loss is None:
            raise ValueError("Snip pruner requires data and loss function.")

        final_scores = []
        for _ in range(16):
            data_ind = np.random.choice(len(self.data_x), 64, replace=False)
            data_x = self.data_x[data_ind]
            data_y = self.data_y[data_ind]
            with tf.GradientTape(watch_accessed_variables=False) as grad_tape:
                for layer in self.layers_to_prune:
                    # Set mask to all ones to evaluate gradient at $c = 1$
                    self._set_mask_on_layer(
                        layer, create_mask(layer.rank_capacity, [], inverted=True)
                    )
                    grad_tape.watch(layer._mask)
                    logits = self.model(data_x)
                    loss = self.loss(data_y, logits)
            grads = grad_tape.gradient(
                loss, [layer._mask for layer in self.layers_to_prune]
            )
            normalized_grads = SnipPruner.normalize(grads)
            final_scores.append(normalized_grads)
        return reduce(lambda x, y: [a + b for a, b in zip(x, y)], final_scores)

    @staticmethod
    def normalize(grads):
        """
        Absolute Value Norm
        Intuition = Large gradients for relaxed binary indicator for weights indicates
        that the weight is important regardless of sign, hence these should correspond
        to higher scores
        :param grads: un-normalized gradients of all the layers
        """
        sum = 0
        for i in range(len(grads)):
            grads[i] = np.abs(np.array(grads[i]))
            sum += np.sum(grads[i])
        for i in range(len(grads)):
            grads[i] /= sum
        return grads
