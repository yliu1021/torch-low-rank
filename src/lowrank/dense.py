from typing import Optional

import numpy as np
import tensorflow as tf
from tensorflow.keras.initializers import GlorotUniform, Zeros, Constant
from tensorflow.keras.layers import Layer
from tensorflow.keras import activations


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

        self.kernel: Optional[tf.Variable] = None
        self.kernel_svd: Optional[tuple[tf.Variable, tf.Variable]] = None
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
        self._rank = new_rank
        if self._rank <= 0 and self._rank != -1:
            raise ValueError(f"Rank must be -1 or positive. Got {self._rank} instead.")
        if self._rank > min(self.num_inputs, self.num_outputs):
            raise ValueError("Rank must be less than min(num inputs, num outputs)")
        if self._rank == -1:
            eff_weight = self.eff_weight()
            if eff_weight is None:
                init = GlorotUniform()
            else:
                init = Constant(eff_weight)
            self.kernel = self.add_weight(
                name="kernel",
                shape=(self.num_inputs, self.num_outputs),
                initializer=init
            )
            self.kernel_svd = None
        else:
            eff_weight = self.eff_weight()
            if eff_weight is None:
                init_u = GlorotUniform()
                init_v = GlorotUniform()
            else:
                u, s, v = np.linalg.svd(eff_weight, full_matrices=False)
                u = u[:, :self.rank]
                s = s[:self.rank] ** 0.5
                v = v[:self.rank, :]
                init_u = Constant(u * s)
                init_v = Constant(s[:, None] * v)
            kernel_u = self.add_weight(
                name="kernel_u",
                shape=(self._rank, self.num_outputs),
                initializer=init_u
            )
            kernel_v = self.add_weight(
                name="kernel_v",
                shape=(self.num_inputs, self._rank),
                initializer=init_v
            )
            self.kernel_svd = (kernel_u, kernel_v)
            self.kernel = None

    def eff_weight(self):
        """
        Return the effective weights of the layer.
        If the layer is in SVD form, return U @ V
        :return: effective weights
        """
        if self.kernel is not None:
            return self.kernel
        if self.kernel_svd is not None:
            kernel_u, kernel_v = self.kernel_svd
            return kernel_v @ kernel_u
        return None

    def build(self, input_shape):
        self.num_inputs = int(input_shape[-1])
        self.set_rank(self._rank)
        self.bias = self.add_weight(
            name="bias",
            shaphe=(self.num_outputs,),
            initializer=Zeros()
        )

    def call(self, inputs, *args, **kwargs):
        pre_act = inputs @ self.eff_weight() + self.bias
        return self.activation(pre_act)
