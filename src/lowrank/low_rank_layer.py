from typing import Dict, Optional, Tuple, Union

import numpy as np
import tensorflow as tf
from tensorflow.keras import activations
from tensorflow.keras.initializers import GlorotUniform
from tensorflow.keras.layers import Layer


class LowRankLayer(Layer):
    def __init__(self, rank: int, activation: Optional[str] = None, **kwargs):
        """
        Creates a low rank layer with a given rank and (optionally) an activation.
        :param rank: the rank of the layer. Specify -1 for full rank
        :param activation: an optional activation to pass. Values can be "relu", "softmax",
        or "sigmoid"
        """
        super().__init__(**kwargs)
        self._rank = rank
        self.activation = activations.get(activation)
        self.num_inputs: Optional[int] = None
        self.num_outputs: Optional[int] = None
        self.kernels: Dict[
            int, Union[tf.Variable, Tuple[tf.Variable, tf.Variable]]
        ] = {}
        self.bias: Optional[tf.Variable] = None
        self.mask = None  # this is set when `set_rank` is called. When `commit_rank` is called,
        # the weights are actually pruned so the mask is set back to None

    @property
    def max_rank(self) -> int:
        return min(self.num_outputs, self.num_inputs)

    @property
    def rank(self) -> int:
        return self._rank

    def set_rank(self, rank_mask: Optional[list[bool]] = None):
        """
        Sets the new rank and creates the appropriate weights (without actually removing singular
        vectors). Call `commit_rank` to actually remove the singular vector.
        :param rank_mask: the rank mask to apply. The SVD is sorted from the largest singular value
        to smallest. Thus, if the first entry of the mask is True, then the largest singular
        value is kept.
        """
        assert self.num_inputs is not None, "Layer needs to be built first"
        if rank_mask is None:
            self._rank = -1
        else:
            if len(rank_mask) > self.max_rank:
                raise ValueError(
                    "Rank mask must have length less than or equal to max rank."
                )
            if len(rank_mask) < self.max_rank:
                rank_mask.extend(
                    [False] * (self.max_rank - len(rank_mask))
                )  # pad mask with 0's
            assert len(rank_mask) == self.max_rank
            self._rank = sum(rank_mask)
        self._allocate_weights(self._rank)
        eff_weights = self.eff_weight()
        if self._rank == -1:
            self.kernels[self._rank].assign(eff_weights)
        else:
            u, s, v = np.linalg.svd(eff_weights, full_matrices=False)
            u = u[:, rank_mask]
            s = s[rank_mask] ** 0.5
            v = v[rank_mask, :]
            kernel_u, kernel_v = self.kernels[self._rank]
            kernel_u.assign(u * s)
            kernel_v.assign(s[:, None] * v)

    def commit_rank(self):
        """
        Commit the pruning action of an ephemeral set rank
        """
        raise NotImplementedError()  # TODO

    def eff_weight(self):
        """
        Return the effective weights of the layer.
        If the layer is in SVD form, return U @ V
        :return: effective weights
        """
        # if there's a mask
        # if self.mask is not None:
        #     return u @ np.diag(self.mask) @ v
        if self._rank not in self.kernels:
            return None
        if self._rank == -1:
            return self.kernels[self._rank]
        else:
            u, v = self.kernels[self._rank]
            return u @ v

    @property
    def trainable_weights(self):
        if self._rank == -1:
            weights = [self.kernels[self._rank]]
        else:
            u, v = self.kernels[self._rank]
            weights = [u, v]
        if self.bias is not None:
            weights.append(self.bias)
        return weights

    def get_config(self):
        config = super().get_config()
        config.update({"rank": self.rank, "activation": self.activation})

    def _allocate_weights(self, rank: int):
        """
        Creates tensorflow weights for a given rank and fills them with glorot uniform.
        """
        if rank in self.kernels:
            return
        if rank == -1:
            self.kernels[rank] = self.add_weight(
                name=f"kernel_{rank}",
                shape=(self.num_inputs, self.num_outputs),
                initializer=GlorotUniform(),
            )
        else:
            self.kernels[rank] = (
                self.add_weight(
                    name=f"kernel_{rank}_u",
                    shape=(self.num_inputs, rank),
                    initializer=GlorotUniform(),
                ),
                self.add_weight(
                    name=f"kernel_{rank}_v",
                    shape=(rank, self.num_outputs),
                    initializer=GlorotUniform(),
                ),
            )
