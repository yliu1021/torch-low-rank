"""
Alignment Pruner Gradient Based
"""
from typing import List

import numpy as np
import tensorflow as tf

from lowrank.pruners import AbstractPrunerBase, create_mask


class AlignmentPrunerGradientBased(AbstractPrunerBase):
    """
    Alignment pruner gradient based scores singular vectors based on how
    much each singular vector perturbs the model gradient from baseline
    """

    def compute_scores(self) -> List[np.ndarray]:
        """
        Score = Magnitude of the vector difference between output of model when passed all 1s
        (with singular vector zeroed out and not)
        Intuition = the singular vectors that change the output vector the most from baseline
        activation are the most important
        """
        assert self.data_x is not None, "Data x is none, cannot infer input shape"
        for layer in self.layers_to_prune:
            layer.mask = np.ones(layer.max_rank())
        self.model._reset_compile_cache()
        scores = []
        data_ind = np.random.choice(len(self.data_x), 64, replace=False)
        data_x = self.data_x[data_ind]
        data_y = self.data_y[data_ind]
        for prune_layer_ind, layer in enumerate(self.layers_to_prune):
            print(f"Pruning low rank layer {prune_layer_ind}")
            layer_scores = []
            print("Getting baseline gradient")
            layer._mask = None  # Set model back to full rank mode
            w = layer.kernel_w  # Get full rank W weight matrix
            with tf.GradientTape(watch_accessed_variables=False) as g:
                g.watch(w)
                y_pred = self.model(data_x)
                baseline_loss = self.loss(data_y, y_pred)
            baseline_gradient = g.gradient(baseline_loss, w)
            u, v = layer.kernel_uv  # Get full u and v matrices - svd of w
            for sv_ind in range(layer.max_rank()):
                # for each singular vector, mask it out and compute new output
                print(f"\rEvaluting singular value {sv_ind}", end="", flush=True)
                u_prime_t = self.remove_row(tf.transpose(u), sv_ind)
                v_prime = self.remove_row(v, sv_ind)
                projected_gradient = tf.add(
                    tf.matmul(
                        baseline_gradient, tf.matmul(v_prime, v_prime, transpose_a=True)
                    ),
                    tf.matmul(
                        tf.matmul(u_prime_t, u_prime_t, transpose_a=True),
                        baseline_gradient,
                    ),
                )
                c = tf.reduce_mean(tf.linalg.diag_part(projected_gradient))
                layer_scores.append(
                    -1
                    * tf.norm(
                        tf.subtract(
                            projected_gradient, tf.scalar_mul(c, baseline_gradient)
                        )
                    )
                )
            print()
            scores.append(np.array(layer_scores))
        return scores

    def remove_row(self, x, sv_ind):
        if sv_ind > 0:
            x_prime = x[0:sv_ind]
            if sv_ind + 1 < x.shape[0]:
                x_prime = tf.concat([x_prime, x[sv_ind + 1 :]], 0)
        else:
            x_prime = x[1:]
        return x_prime
