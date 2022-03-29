"""
SNIP SV Pruner - calculates Delta L instead of gradient estimate
"""

from lowrank.pruners import AbstractPrunerBase, create_mask
import tensorflow as tf
import numpy as np

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

        with tf.GradientTape(watch_accessed_variables=False) as grad_tape:
            for layer in self.layers_to_prune:
                # Set mask to all ones to evaluate gradient at $c = 1$
                self._set_mask_on_layer(layer, create_mask(layer.rank_capacity, [], inverted=True))
                grad_tape.watch(layer._mask)
                logits = self.model(self.data_x)
                loss = self.loss(self.data_y, logits)
        grads = grad_tape.gradient(loss, [layer._mask for layer in self.layers_to_prune])
        normalized_grads = SnipPruner.normalize(grads)
        return normalized_grads

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
