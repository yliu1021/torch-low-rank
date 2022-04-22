"""
Alignment Pruner (Defined in overleaf)
"""
from typing import List

import numpy as np
from tensorflow.keras.metrics import KLDivergence

from lowrank.pruners import AbstractPrunerBase, create_mask


class AlignmentPruner(AbstractPrunerBase):
    """
    Alignment pruners scores singular vectors based on how
    much each singular vector perturbs the model output from
    the baseline
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
            layer.mask = np.ones(layer.max_rank)
        self.model._reset_compile_cache()
        scores = []
        data_ind = np.random.choice(len(self.data_x), 64, replace=False)
        data_x = self.data_x[data_ind]
        print("Getting baseline output")
        baseline_output = self.model(data_x)
        for layer_ind, layer in enumerate(self.layers_to_prune):
            print(f"Pruning layer {layer_ind}")
            layer_scores = []
            for sv_ind in range(layer.max_rank):
                # for each singular vector, mask it out and compute new output
                print(f"\rEvaluting singular value {sv_ind}", end="", flush=True)
                new_mask = np.ones(layer.max_rank)
                new_mask[sv_ind] = 0
                layer.mask = new_mask
                self.model._reset_compile_cache()
                new_output = self.model(data_x)
                divergence = KLDivergence()
                divergence.update_state(baseline_output, new_output)
                layer_scores.append(divergence.result().numpy())
                layer.mask = np.ones(layer.max_rank)
                self.model._reset_compile_cache()
            print()
            scores.append(np.array(layer_scores))
        return scores
