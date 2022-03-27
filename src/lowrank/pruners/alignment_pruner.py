"""
Alignment Pruner (Defined in overleaf)
"""

import tensorflow as tf
import numpy as np

from lowrank.pruners import AbstractPrunerBase, create_mask


class AlignmentPruner(AbstractPrunerBase):
    """
    Alignment pruners scores singular vectors based on how
    much each singular vector perturbs the model output from
    the baseline
    """

    def compute_scores(self) -> "list[list[int | float]]":
        """
        Score = Magnitude of the vector difference between output of model when passed all 1s
        (with singular vector zeroed out and not)
        Intuition = the singular vectors that change the output vector the most from baseline
        activation are the most important
        """
        assert self.data_x is not None, "Data x is none, cannot infer input shape"
        scores = []
        for layer_ind, layer in enumerate(self.layers_to_prune):
            print(f"Pruning layer: {layer_ind}")
            layer_scores = []
            self.set_mask_on_layer(layer, create_mask(layer.rank_capacity, []))
            all_ones_input = tf.convert_to_tensor(
                [tf.ones(self.data_x.shape[1:])], dtype=np.float64
            )
            baseline_output_activation = self.model.call(all_ones_input)
            for i in range(layer.rank_capacity):
                self.set_mask_on_layer(layer, create_mask(layer.rank_capacity, [i]))
                sv_output_activation = self.model.call(all_ones_input)
                layer_scores.append(
                    kl_divergence(baseline_output_activation, sv_output_activation)
                )
            self.set_mask_on_layer(
                layer, create_mask(layer.rank_capacity, [], inverted=True)
            )
            scores.append(layer_scores)
        return scores

def kl_divergence(p, q):
    """
    Safe implementation of KL Divergence, using https://stats.stackexchange.com/questions/362860/kl-divergence-between-which-distributions-could-be-infinity
    :param p: Target Distribution
    :param q: Approximate Distribution
    """
    p = np.array(p[0])
    q = np.array(q[0])
    if len(p) != len(q):
        raise Exception("Length of the two distibutions must be identical")
    kl = 0
    for x in range(len(p)):
        if p[x] == 0:
            continue
        elif q[x] == 0:
            kl = float('inf')
            break 
        kl += p[x] * np.log2(p[x] / q[x])
    return kl
        
