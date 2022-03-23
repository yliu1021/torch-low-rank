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
        if rank == -1:
            self._mask: Optional[np.ndarray] = None
        else:
            self._mask: Optional[np.ndarray] = np.array([True] * rank)
        self.activation = activations.get(activation)
        self.num_inputs: Optional[int] = None
        self.num_outputs: Optional[int] = None
        self.kernels: Dict[
            int, Union[tf.Variable, Tuple[tf.Variable, tf.Variable]]
        ] = {}
        self.bias: Optional[tf.Variable] = None

    @property
    def max_rank(self) -> int:
        return min(self.num_outputs, self.num_inputs)

    @property
    def rank_capacity(self) -> Optional[int]:
        if self._mask is None:
            return None
        return len(self._mask)

    @property
    def rank(self) -> int:
        if self._mask is None:
            return self.max_rank
        return sum(self._mask)

    def set_mask(self, new_mask: 'list[bool]'):
        if len(new_mask) != len(self._mask):
            raise ValueError("New mask must have the same size")
        self._mask = np.array(new_mask)

    def set_rank_capacity(self, capacity: Optional[int] = None):
        """
        Sets the new rank and creates the appropriate weights (without actually removing singular
        vectors). Call `commit_rank` to actually remove the singular vector.
        :param capacity: The capacity
        """
        assert self.num_inputs is not None, "Layer needs to be built first"
        if self.rank_capacity == capacity:
            raise ValueError("Setting capacity to current capacity")
        if capacity > self.max_rank:
            raise ValueError("Rank capacity must be less than or equal to max rank.")
        eff_weights = self.eff_weight()
        if capacity is None:
            self._mask = None
            self._allocate_weights(-1)
            self.kernels[-1].assign(eff_weights)
            return
        self._mask = [True] * capacity
        self._allocate_weights(self.rank_capacity)
        u, s, v = np.linalg.svd(eff_weights, full_matrices=False)
        u = u[:, : self.rank_capacity]
        s = s[: self.rank_capacity] ** 0.5
        v = v[: self.rank_capacity, :]
        kernel_u, kernel_v = self.kernels[self.rank_capacity]
        kernel_u.assign(u * s)
        kernel_v.assign(s[:, None] * v)

    def squeeze_rank_capacity(self):
        """
        Removes unneeded singular vectors. I.e. removes singular vectors that are masked out
        """
        if self.rank_capacity is None:
            # rank -1 layers cannot be squeezed because we have no mask
            return
        if all(self._mask):
            return
        self.set_rank_capacity(self.rank)

    def eff_weight(self) -> tf.Variable:
        """
        Return the effective weights of the layer.
        If the layer is in SVD form, return U @ V
        :return: effective weights
        """
        if self._mask is None:
            # we can't mask here
            return self.kernels[-1]
        else:
            u, v = self.kernels[self.rank_capacity]
            return u @ np.diag(self._mask) @ v

    @property
    def trainable_weights(self):
        if self.rank_capacity == None:
            weights = [self.kernels[-1]]
        else:
            u, v = self.kernels[self.rank_capacity]
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
            return
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
