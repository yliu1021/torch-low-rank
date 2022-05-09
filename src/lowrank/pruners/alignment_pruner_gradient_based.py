"""
Alignment Pruner Gradient Based
"""
from typing import List

import numpy as np
import torch

from lowrank.pruners import AbstractPrunerBase


class AlignmentPrunerGradientBased(AbstractPrunerBase):
    """
    Alignment pruners scores singular vectors based on how
    much each of the gradient is preserved by projecting on
    to a given singular vector 
    """

    def compute_scores(self) -> List[np.ndarray]:
        """
        Score = Magnitude of difference of projected gradient with & without this change from actual
        gradients
        Intuition = the singular vectors that preserve the gradient the best are the best
        """
        # Sets mask to all ones to trigger svd
        print("Setting mask to trigger SVD (if needed)")
        for layer in self.layers_to_prune:
            if (
                layer.mask is None
            ):  # if mask already set, do not want to overwrite it (needed for iterative pruning)
                layer.mask = torch.ones(layer.max_rank()).to(self.device)

        scores = []

        # Baseline Gradient
        print("Getting baseline gradients")
        X, y = next(iter(self.dataloader))
        X = X.to(self.device)
        y = y.to(self.device)
        loss = self.loss(self.model(X), y)
        loss.backward()
        baseline_gradients = []
        for layer in self.layers_to_prune:
            if not layer.svd_masking_mode: # if mask is already set, don't overwrite it
                layer.mask = torch.ones(layer.max_rank()).to(self.device)
            baseline_gradients.append(layer.eff_weights.grad) # this is = dl/dw vt v + u ut dl/dw

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

                # Compute network gradient -> determine score
                u_prime_t = self.remove_row(layer.kernel_u.T, sv_ind)
                v_prime = self.remove_row(layer.kernel_v, sv_ind)
                new_gradient = baseline_gradients[layer_ind] @ v_prime.T @ v_prime + u_prime_t.T @ u_prime_t @ baseline_gradients[layer_ind]
                layer_scores.append(
                    torch.norm(
                       torch.subtract(new_gradient, baseline_gradients[layer_ind]).detach().cpu()
                    )
                )

            print()
            scores.append(np.array(layer_scores))

        return scores


    def remove_row(self, x, sv_ind):
        if sv_ind > 0:
            x_prime = x[0:sv_ind]
            if sv_ind + 1 < x.shape[0]:
                x_prime = torch.concat([x_prime, x[sv_ind+1:]], 0)
        else:
            x_prime = x[1:]
        return x_prime