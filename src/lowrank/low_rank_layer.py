from typing import Dict, Optional, Tuple, Union

import numpy as np
import tensorflow as tf
from tensorflow.keras import activations, regularizers
from tensorflow.keras.initializers import GlorotUniform
from tensorflow.keras.layers import Layer


class LowRankLayer(Layer):
    def __init__(
        self,
        rank: int,
        activation: Optional[str] = None,
        weight_decay: float = 1e-4,
        **kwargs,
    ):
        """
        Creates a low rank layer with a given rank and (optionally) an activation.
        :param rank: the rank of the layer. Specify -1 for full rank
        :param activation: an optional activation to pass. Values can be "relu", "softmax",
        :param weight_decay: the L2 reg term
        or "sigmoid"
        """
        super().__init__(**kwargs)
        if rank == -1:
            self._mask: Optional[tf.Variable] = None
        else:
            self._mask: Optional[tf.Variable] = tf.Variable(
                [1.0] * rank, trainable=False
            )
        self.activation = activations.get(activation)
        self.num_inputs: Optional[int] = None
        self.num_outputs: Optional[int] = None
        self.kernel_w: Optional[tf.Variable] = None
        self.kernel_uv: Optional[Tuple[tf.Variable, tf.Variable]] = None
        self.bias: Optional[tf.Variable] = None
        self.weight_decay = weight_decay

    @property
    def max_rank(self) -> int:
        return min(self.num_outputs, self.num_inputs)

    @property
    def rank_capacity(self) -> Optional[int]:
        # DEPRECATED
        raise RuntimeError("Rank capacity is deprecated")
        if self._mask is None:
            return None
        if len(self._mask.shape) == 1:
            return self._mask.shape[0]
        else:
            return None

    @property
    def rank(self) -> int:
        if self._mask is None:
            return self.max_rank
        if len(self._mask.shape) == 1:
            return int(sum(self._mask.numpy()))
        else:
            return self.max_rank

    def set_mask(self, new_mask: tf.Variable):
        if self._mask is None:
            self._mask = tf.Variable(new_mask, trainable=False, dtype=tf.float32)
        else:
            self._mask = tf.Variable(new_mask, trainable=False, dtype=tf.float32)

    def set_rank_capacity(self, capacity: Optional[int] = None):
        """
        Performs a SVD and shrink the size of the U and V matrices.
        :param capacity: The capacity
        """
        if capacity is not None and capacity != self.max_rank:
            raise ValueError("Setting rank capacity to no full rank is deprecated")
        assert self.num_inputs is not None, "Layer needs to be built first"
        if self.rank_capacity == capacity:
            raise ValueError("Setting capacity to current capacity")
        if capacity > self.max_rank:
            raise ValueError("Rank capacity must be less than or equal to max rank.")
        eff_weights = self.eff_weight()
        if capacity is None:
            self._mask = None
            self._allocate_weights(-1)
            self.kernel_w.assign(eff_weights)
            return
        self._mask = tf.Variable([1.0] * capacity, trainable=False)
        self._allocate_weights(self.rank_capacity)
        u, s, v = np.linalg.svd(eff_weights, full_matrices=False)
        u = u[:, : self.rank_capacity]
        s = s[: self.rank_capacity] ** 0.5
        v = v[: self.rank_capacity, :]
        kernel_u, kernel_v = self.kernel_uv
        kernel_u.assign(u * s)
        kernel_v.assign(s[:, None] * v)

    def squeeze_rank_capacity(self):
        """
        Removes unneeded singular vectors. I.e. removes singular vectors that are masked out
        """
        # DEPRECATED
        raise RuntimeError("Squeeze rank capacity is deprecated")
        if self.rank_capacity is None:
            # rank -1 layers cannot be squeezed because we have no mask
            return
        if np.array(self._mask > 0).all():
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
            return self.kernel_w
        elif len(self._mask.shape) == 1:
            u, v = self.kernel_uv
            return u @ tf.linalg.diag(self._mask) @ v
        else:
            return self.kernel_w * self._mask

    @property
    def trainable_weights(self):
        if self._mask is None:
            weights = [self.kernel_w]
        else:
            u, v = self.kernel_uv
            weights = [u, v]
        if self.bias is not None:
            weights.append(self.bias)
        return weights

    def get_config(self):
        config = super().get_config()
        config.update({"rank": self.rank, "activation": self.activation})
        return config

    def _allocate_weights(self, rank: int):
        """
        Creates tensorflow weights for a given rank and fills them with glorot uniform.
        """
        if rank == -1 and self.kernel_w is None:
            self.kernel_w = self.add_weight(
                name=f"kernel_{rank}",
                shape=(self.num_inputs, self.num_outputs),
                initializer=GlorotUniform(),
                regularizer=regularizers.l2(self.weight_decay),
            )
            return
        if self.kernel_uv is not None:
            return
        u = self.add_weight(
            name=f"kernel_{rank}_u",
            shape=(self.num_inputs, rank),
            initializer=GlorotUniform(),
            # regularizer=regularizers.l2(self.weight_decay),
        )
        v = self.add_weight(
            name=f"kernel_{rank}_v",
            shape=(rank, self.num_outputs),
            initializer=GlorotUniform(),
            # regularizer=regularizers.l2(self.weight_decay),
        )
        self.add_loss(self.weight_decay * tf.norm(u @ v))
        self.kernel_uv = (u, v)
