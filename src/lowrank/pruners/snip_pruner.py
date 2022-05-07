"""
SNIP SV Pruner - calculates Delta L instead of gradient estimate
"""
from functools import reduce
from typing import List

import numpy as np
import tensorflow as tf

from lowrank.pruners import AbstractPrunerBase, create_mask


class SnipPruner(AbstractPrunerBase):
    """
    Class for SNIP Pruner
    Implements compute score to score singular vectors using SNIP Method.
    """

    def compute_scores(self) -> List[np.ndarray]:
        """
        Score = loss if masking out the singular vector
        Intuition = if loss when masking out the singular vector is high,
        then the singular vector must be important.
        """
        if self.dataloader is None or self.loss is None:
            raise ValueError("Snip pruner requires data and loss function.")
        for layer in self.layers_to_prune:
            layer.mask = np.ones(layer.max_rank)
        self.model._reset_compile_cache()
        grads_tot = []
        for _ in range(16):
            data_ind = np.random.choice(len(self.data_x), 64, replace=False)
            data_x = self.data_x[data_ind]
            data_y = self.data_y[data_ind]
            with tf.GradientTape(watch_accessed_variables=False) as grad_tape:
                for layer in self.layers_to_prune:
                    # Set mask to all ones to evaluate gradient at $c = 1$
                    grad_tape.watch(layer.mask)
                    output = self.model(data_x)
                    loss = self.loss(data_y, output)
            grads = grad_tape.gradient(
                loss, [layer.mask for layer in self.layers_to_prune]
            )
            grads_tot.append([np.array(grad) for grad in grads])
        final_grads = reduce(lambda x, y: [a + b for a, b in zip(x, y)], grads_tot)
        return final_grads
