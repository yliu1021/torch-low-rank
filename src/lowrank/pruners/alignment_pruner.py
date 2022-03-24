"""
Alignment Pruner (Defined in overleaf)
"""

from numpy import float64
import tensorflow as tf
from lowrank.pruners import create_mask, AbstractPrunerBase

class AlignmentPruner(AbstractPrunerBase):
    """
    Alignment pruners scores singular vectors based on how
    much each singular vector perturbs the model output from
    the baseline
    """
    def compute_scores(self) -> 'list[list[int | float]]':
        """
        Score = Magnitude of the vector difference between output of model when passed all 1s
        (with singular vector zeroed out and not)
        Intuition = the singular vectors that change the output vector the most from baseline
        activation are the most important
        """
        scores = []
        for layer in self.layers_to_prune:
            layer_scores = []
            self.set_mask_on_layer(layer, create_mask(layer.rank_capacity, []))
            all_ones_input = tf.convert_to_tensor([tf.ones(self.data_x.shape[1:])], dtype=float64)
            baseline_output_activation = self.model.call(all_ones_input)
            for i in range(layer.rank_capacity):
                self.set_mask_on_layer(layer, create_mask(layer.rank_capacity, [i]))
                sv_output_activation = self.model.call(all_ones_input)
                layer_scores.append(float(tf.norm(
                    tf.math.subtract(baseline_output_activation, sv_output_activation)
                )))
            scores.append(layer_scores)
        return scores
