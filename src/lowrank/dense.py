from typing import Optional, Union

import numpy as np
import tensorflow as tf
from tensorflow.keras import activations
from tensorflow.keras.initializers import Constant, GlorotUniform, Zeros
from tensorflow.keras.layers import Layer


class LRDense(Layer):
    def __init__(self, num_outputs: int, rank: int, activation: Optional[str] = None):
        """
        Creates a low rank layer with a given rank and (optionally) an activation.
        :param num_outputs: the number of outputs for this layer
        :param rank: the rank of the layer. Specify -1 for full rank
        :param activation: an optional activation to pass. Values can be "relu", "softmax",
        or "sigmoid"
        """
        super().__init__()
        self.num_inputs: Optional[int] = None
        self.num_outputs = num_outputs
        self._rank = rank
        self.activation = activations.get(activation)

        self.kernels: dict[int, Union[tf.Variable, tuple[tf.Variable, tf.Variable]]] = {}
        # self.kernel: Optional[tf.Variable] = None
        # self.kernel_svd: Optional[tuple[tf.Variable, tf.Variable]] = None
        self.bias: Optional[tf.Variable] = None

    @property
    def rank(self) -> int:
        return self._rank

    def set_rank(self, new_rank: int):
        """
        Sets the new rank and creates the appropriate weights
        :param new_rank: the new rank to set to. Set to -1 for full rank
        :return:
        """
        assert self.num_inputs is not None, "Layer needs to be built first"
        eff_weights = self.eff_weight()
        self._rank = new_rank
        if self._rank <= 0 and self._rank != -1:
            raise ValueError(f"Rank must be -1 or positive. Got {self._rank} instead.")
        if self._rank > min(self.num_inputs, self.num_outputs):
            raise ValueError("Rank must be less than min(num inputs, num outputs)")
        self._create_weights(self._rank)
        if self._rank == -1:
            self.kernels[self._rank].assign(eff_weights)
        else:
            u, s, v = np.linalg.svd(eff_weights, full_matrices=False)
            u = u[:, : self.rank]
            s = s[: self.rank] ** 0.5
            v = v[: self.rank, :]
            kernel_u, kernel_v = self.kernels[self._rank]
            kernel_u.assign(u * s)
            kernel_v.assign(s[:, None] * v)

    def eff_weight(self):
        """
        Return the effective weights of the layer.
        If the layer is in SVD form, return U @ V
        :return: effective weights
        """
        if self._rank not in self.kernels:
            return None
        if self._rank == -1:
            return self.kernels[self._rank]
        else:
            u, v = self.kernels[self._rank]
            return u @ v

    def build(self, input_shape):
        self.num_inputs = int(input_shape[-1])
        self._create_weights(self._rank)
        self.bias = self.add_weight(
            name="bias", shape=(self.num_outputs,), initializer=Zeros()
        )

    def call(self, inputs, *args, **kwargs):
        pre_act = inputs @ self.eff_weight() + self.bias
        return self.activation(pre_act)

    def _create_weights(self, rank: int):
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
                    initializer=GlorotUniform()),
                self.add_weight(
                    name=f"kernel_{rank}_v",
                    shape=(rank, self.num_outputs),
                    initializer=GlorotUniform()),
            )
