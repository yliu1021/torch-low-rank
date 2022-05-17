"""
Gradient Descent Pruner Loss Based
"""
from typing import List

import numpy as np
import torch
from torch import nn, optim

from lowrank_experiments import trainer
from lowrank.low_rank_layer import LowRankLayer
from lowrank.pruners import AbstractPrunerBase


class GDPrunerLossBased(AbstractPrunerBase):
    """
    Gradient Descent Pruner uses gradient descent to optimize the masks
    """

    def compute_scores(self, target_sparsity=None) -> List[np.ndarray]:
        print("Setting mask to trigger SVD (if needed)")
        for layer in self.layers_to_prune:
            if layer.mask is None:
                # if mask already set, do not want to overwrite it (needed for iterative pruning)
                layer.mask = torch.ones(layer.max_rank()).to(self.device)
            layer.additional_mask = layer.mask.data

        # if sparisty not explicity specified then use sparsity for pruner
        if target_sparsity == None:
            target_sparsity = self.sparsity

        # Mask training mode -> on
        self.additional_mask_train_mode(self.model, turn_on=True)

        # Define new loss function with regularization
        def new_loss(output, target):
            return self.loss(output, target) + self.sparsity_bonus * (target_sparsity - self.effective_sparsity())

        opt = optim.SGD(
            self.model.parameters(), lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay
        )
        for _ in range(2):
            trainer.train(self.model, self.dataloader, new_loss, opt, self.device, None, None)
            print("Effective Sparsity > Target Sparsity = ", self.effective_sparsity() > target_sparsity, self.effective_sparsity(), target_sparsity) 
            if self.effective_sparsity() >= target_sparsity:
                break
        
        for layer in self.layers_to_prune:
            print(layer.additional_mask)
            
        print("Effective Sparsity > Target Sparsity = ", self.effective_sparsity() > target_sparsity, self.effective_sparsity(), target_sparsity) 
    
        # Mask training mode -> off
        self.additional_mask_train_mode(self.model, turn_on=False)
        
        # Extract scores from additional mask
        scores = []
        for layer in self.layers_to_prune:
            scores.append(np.array(torch.abs(layer.additional_mask).detach().cpu()))
        return scores

    def additional_mask_train_mode(self, parent_module: nn.Module, turn_on: bool):
        for child in parent_module.children():
            if len(list(child.children())) == 0:
                if isinstance(child, LowRankLayer):
                    for name, param in child.named_parameters():
                        if "additional_mask" in name:
                            param.requires_grad = turn_on
                        else:
                            param.requires_grad = not turn_on
                else:
                    for param in child.parameters():
                        param.requires_grad = not turn_on
            else:
               self.additional_mask_train_mode(child, turn_on)