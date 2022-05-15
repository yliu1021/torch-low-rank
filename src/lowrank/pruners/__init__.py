"""
Pruner Base Class Implementation and other useful package wide code
"""
import enum
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from torch import nn, optim

from lowrank.low_rank_layer import LowRankLayer

import trainer


class PruningScope(enum.Enum):
    """
    Pruning Scope determines how to use scores to rank singular vectors and generate mask.
    Global ranks globally, Local ranks locally
    """

    GLOBAL = enum.auto()  # global pruning will score all ranks from all layers together
    LOCAL = enum.auto()  # local pruning will treat each layer independently


class AbstractPrunerBase:
    """
    Pruners take a model, and upon examining its effective weights, computes rank masks for
    each layer
    """

    def __init__(
        self,
        device,
        model: nn.Module,
        scope: PruningScope,
        sparsity: float,
        dataloader=None,
        opt=None,
        batch_size: int = 64,
        loss=None,
        prune_iterations=1,
    ):
        self.device = device
        self.model = model
        self.scope = scope
        if sparsity < 0 or sparsity > 1:
            raise ValueError("Sparsity must be in the range [0, 1]")
        self.sparsity = sparsity
        self.dataloader = dataloader
        self.batch_size = batch_size
        self.loss = loss
        self.layers_to_prune: List[LowRankLayer] = list(
            filter(lambda x: isinstance(x, LowRankLayer), list(self.model.modules()))
        )
        self.prune_iterations = prune_iterations
        self.opt = opt

    def compute_scores(self, target_sparsity) -> List[np.ndarray]:
        """
        Computes and returns scores for the singular vectors in each layer.
        - High Score = Important Singular Vector
        - Low Score = Unimportant Singular Vector
        """
        raise NotImplementedError("Must be called on a subclass of Pruner")

    def prune(self):
        """
        Calls the `compute_mask` method and actually sets the ranks
        """

        # Iterative pruning done by increasing desired sparsity every iteration
        # sparsity_per_iteration = np.cumsum(
        #     [self.sparsity / self.prune_iterations] * self.prune_iterations
        # )
        loss_fn = nn.CrossEntropyLoss()
        p = 15
        sparsities = np.linspace(0 ** p, self.sparsity ** p, num=self.prune_iterations+1)[1:] ** (1/p)
        print("sparsities", sparsities)
        for prune_iteration, sparsity in enumerate(sparsities):
            print(f"Prune iteration {prune_iteration+1} / {self.prune_iterations}")
            self.sparsity = sparsity
            masks = self._compute_masks()
            if len(masks) != len(self.layers_to_prune):
                raise ValueError("Computed mask does not match length of model layers")
            for mask, layer in zip(masks, self.layers_to_prune):
                layer.mask = mask
            print(f"sparsity: {sparsity:.2f}")
            trainer.train(self.model, self.dataloader, loss_fn, self.opt, device=self.device)

        # retrain batch norm 
        def batch_norm_mode(parent_model: nn.Module, turn_on: bool):
            for child in parent_model.children():
                if len(list(child.children())) == 0:
                    if not isinstance(child, nn.BatchNorm2d):
                        for param in child.parameters():
                            param.requires_grad = not turn_on
                else:
                    batch_norm_mode(child, turn_on)
        print("Retrain batch norm")
        batch_norm_mode(self.model, True)
        for _ in range(2):
            trainer.train(self.model, self.dataloader, loss_fn, self.opt, device=self.device)
        batch_norm_mode(self.model, False)

    def _compute_masks(self):
        """
        Create masks for the pruning method.
        Calls compute scores which is implemented by the subclass overriding the base Pruner class.
        Creates mask to drop lowest scores in accordance with sparsity ratio.
        """
        # list of ndarrays, each corresponding to each layer
        scores = self.compute_scores()
        assert len(scores) == len(
            self.layers_to_prune
        ), "Number of scores should equal number of layers we're trying to prune"
        if self.scope == PruningScope.LOCAL:
            thresholds = []
            for i in range(len(self.layers_to_prune)):
                sorted_layer_scores = sorted(scores[i].flatten())
                num_to_drop = int(len(sorted_layer_scores) * self.sparsity)
                thresholds.append(sorted_layer_scores[num_to_drop])
        elif self.scope == PruningScope.GLOBAL:
            flattened_sorted_scores = sorted(
                np.concatenate([score.flatten() for score in scores])
            )
            num_to_drop = int(len(flattened_sorted_scores) * self.sparsity)
            thresholds = [flattened_sorted_scores[num_to_drop]] * len(
                self.layers_to_prune
            )
        else:
            raise NotImplementedError(f"{self.scope} is not supported yet.")
        masks = [
            torch.tensor((score >= threshold).astype(np.float32), device=self.device)
            for score, threshold in zip(scores, thresholds)
        ]
        return masks
    
    def effective_sparsity(self):
        return self.num_eff_params(self.model) / self.num_params_unpruned(self.model)

    def num_params_unpruned(self, parent_module):
        total_params = 0
        if len(list(parent_module.children())) == 0:
            if isinstance(parent_module, LowRankLayer):
                total_params = parent_module.kernel_w.numel()
            else:
                total_params = 0
                for param in parent_module.parameters():
                    total_params += param.numel()
        else:
            total_params = sum([self.num_eff_params(module) for module in parent_module.children()])
        return total_params

    def num_eff_params(self, parent_module: nn.Module):
        total_params = 0
        if len(list(parent_module.children())) == 0:
            if isinstance(parent_module, LowRankLayer):
                total_params = parent_module.num_effective_params()
            else:
                total_params = 0
                for param in parent_module.parameters():
                    total_params += param.numel()
        else:
            total_params = sum([self.num_eff_params(module) for module in parent_module.children()])
        return total_params

def create_mask(
    length: int,
    indices: List[int],
    inverted: bool = False,
):
    """
    Helper function that creates mask given
    :param length: Length of bool vector
    :param indices: Indices to set to true (if inverted=False i.e. default) and rest set to false
    :param inverted: set to false (default) default behavior, set to true - element-wise not
    :returns: bool vector with only variables at indices set to true if inverted=False (default)
    """
    mask = [float(x in indices) for x in range(length)]
    if inverted:
        mask = [(1 - x) for x in mask]
    return np.array(mask)