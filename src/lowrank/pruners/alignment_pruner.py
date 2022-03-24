"""
Alignment Pruner (Defined in overleaf)
"""

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
            baseline_output_activation = self.model.call(tf.ones(self.model.input_shape))
            for i in range(layer.rank_capacity):
                self.set_mask_on_layer(layer, create_mask(layer.rank_capacity, [i]))
                sv_output_activation = self.model.call(tf.ones(self.model.input_shape))
                layer_scores.append(tf.norm(
                    tf.math.subtract(baseline_output_activation, sv_output_activation)
                ))
            scores.append(layer_scores)
        return scores
