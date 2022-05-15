"""
Gradient Descent Pruner Loss Based
"""
from typing import List

import numpy as np
import torch

from lowrank_experiments import trainer
from lowrank.low_rank_layer import LowRankLayer
from lowrank.pruners import AbstractPrunerBase


class GDPrunerLossBased(AbstractPrunerBase):
    """
    Gradient Descent Pruner uses gradient descent to optimize the masks
    """

    def compute_scores(self, target_sparsity=None) -> List[np.ndarray]:
        
        # if sparisty not explicity specified then use sparsity for pruner
        if target_sparsity == None:
            target_sparsity = self.sparsity

        # Mask training mode -> on
        self.additional_mask_train_mode(turn_on=True)

        # Define new loss function with regularization
        def new_loss(output, target):
            return self.loss(output, target) - self.sparsity_bonus * self.effective_sparsity()

        for _ in range(100):
            trainer.train(self.model, self.dataloader, new_loss, self.opt, self.device, None, None)
            if self.effective_sparsity() >= target_sparsity:
                break

        print("Effective Sparsity > Target Sparsity = ", self.effective_sparsity() > target_sparsity, self.effective_sparsity, target_sparsity) 
    
        # Mask training mode -> off
        self.additional_mask_train_mode(turn_on=False)
        
        # Extract scores from additional mask
        scores = []
        for layer in self.layers_to_prune:
            scores.append(np.array(torch.abs(layer.additional_mask).detach().cpu()))
        return scores

    def additional_mask_train_mode(self, turn_on: bool):
        for child in self.model.children():
            if len(list(child.children())) == 0:
                if not isinstance(child, LowRankLayer):
                    for name, param in child.named_parameters():
                        print(name)
                        if "additional_mask" in name:
                            param.requires_grad = turn_on
                        else:
                            param.requires_grad = not turn_on
            else:
               self.additional_mask_train_mode(child, turn_on)