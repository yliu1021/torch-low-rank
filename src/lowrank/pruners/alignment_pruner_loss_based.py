"""
Alignment Pruner Loss Based
"""
from typing import List
import torch
import numpy as np
from lowrank.pruners import AbstractPrunerBase


class AlignmentPrunerLossBased(AbstractPrunerBase):
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
        
        # Sets mask to all ones to trigger svd
        print("Setting mask to trigger SVD (if needed)")
        for layer in self.layers_to_prune:
            if layer.mask is None: # if mask already set, do not want to overwrite it (needed for iterative pruning)
                layer.mask = np.ones(layer.max_rank, dtype=float)

        scores = []
        data_ind = np.random.choice(len(self.data_x), 64, replace=False)
        data_x = self.data_x[data_ind]

        print("Getting baseline output")
        baseline_output = self.model(data_x)

        for layer_ind, layer in enumerate(self.layers_to_prune):
            print(f"Pruning low rank layer {layer_ind}")
            
            layer_scores = []
            for sv_ind in range(layer.max_rank):
                # For each singular vector, mask it out and compute new output
                print(f"\rEvaluting singular value {sv_ind}", end="", flush=True)
                
                # Compute and apply additional mask
                additional_mask = np.ones(layer.max_rank, dtype=float)
                additional_mask[sv_ind] = 0
                layer.additional_mask = additional_mask

                # Compute network output -> determine score
                new_output = self.model(data_x)
                layer_scores.append(
                    torch.norm(torch.subtract(baseline_output, new_output))
                )

                # Clean up additional mask
                layer.additional_mask = None

            scores.append(np.array(layer_scores))

        return scores
