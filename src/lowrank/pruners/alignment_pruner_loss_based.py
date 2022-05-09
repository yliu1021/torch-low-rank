"""
Alignment Pruner Loss Based
"""
from typing import List

import numpy as np
import torch

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
        # Sets mask to all ones to trigger svd
        print("Setting mask to trigger SVD (if needed)")
        for layer in self.layers_to_prune:
            if (
                layer.mask is None
            ):  # if mask already set, do not want to overwrite it (needed for iterative pruning)
                layer.mask = torch.ones(layer.max_rank()).to(self.device)

        scores = []

        # Baseline Output
        print("Getting baseline output")
        X, _ = next(iter(self.dataloader))
        X = X.to(self.device)
        baseline_output = self.model(X)

        for layer_ind, layer in enumerate(self.layers_to_prune):
            print(f"Pruning low rank layer {layer_ind}")

            layer_scores = []
            for sv_ind in range(layer.max_rank()):
                # For each singular vector, mask it out and compute new output
                print(f"\rEvaluting singular value {sv_ind}", end="", flush=True)

                # Compute and apply additional mask
                additional_mask = torch.ones(layer.max_rank()).to(self.device)
                additional_mask[sv_ind] = 0
                layer.additional_mask = additional_mask

                # Compute network output -> determine score
                new_output = self.model(X)
                layer_scores.append(
                    torch.norm(
                        torch.subtract(baseline_output, new_output).detach().cpu()
                    )
                )

                # Clean up additional mask
                layer.additional_mask = None
            print()

            scores.append(np.array(layer_scores))

        return scores
