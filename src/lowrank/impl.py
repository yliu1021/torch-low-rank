from typing import Optional, Tuple, Union

import tensorflow as tf
from tensorflow.keras.initializers import Zeros

from .low_rank_layer import LowRankLayer


class LRDense(LowRankLayer):
    def __init__(self, num_outputs: int, rank: int, activation: Optional[str] = None):
        """
        Creates a low rank dense layer with a given rank and (optionally) an activation.
        Args:
            num_outputs (int): the number of outputs for this layer
            rank (int): the rank of the layer. Specify -1 for full rank
            activation (:obj:`str`, optional): an optional activation to pass. Values can be "relu", "softmax",
            or "sigmoid"
        """
        super().__init__(rank, activation)
        self.num_outputs = num_outputs

    def build(self, input_shape):
        self.num_inputs = int(input_shape[-1])
        self._create_weights(self._rank)
        self.bias = self.add_weight(
            name="bias", shape=(self.num_outputs,), initializer=Zeros()
        )

    def call(self, inputs, *args, **kwargs):
        """
        Like pytorch forward function
        """
        if self._rank == -1:
            pre_act = inputs @ self.kernels[self._rank] + self.bias
        else:
            u, v = self.kernels[self._rank]
            pre_act = inputs @ u @ v + self.bias
        return self.activation(pre_act)


class LRConv2D(LowRankLayer):
    def __init__(
        self,
        filters: int,
        kernel_size: Union[int, Tuple[int, int]],
        rank: int,
        strides: int = 1,
        activation: Optional[str] = None,
        padding: str = "same",
    ):
        super().__init__(rank, activation)
        self.num_outputs = filters
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if len(kernel_size) != 2:
            raise ValueError("Kernel size must be a list/tuple of 2 elements")
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding.upper()

    def build(self, input_shape):
        h, w = self.kernel_size
        num_in_channels = input_shape[-1]
        self.num_inputs = h * w * num_in_channels
        self._create_weights(self._rank)
        self.bias = self.add_weight(
            name="bias", shape=(self.num_outputs,), initializer=Zeros()
        )

    def call(self, inputs, *args, **kwargs):
        """
        Like pytorch forward function
        """
        weights = self.eff_weight()
        h, w = self.kernel_size
        num_in_channels = self.num_inputs // (h * w)
        weights = tf.reshape(weights, shape=(h, w, num_in_channels, self.num_outputs))
        pre_act = tf.nn.conv2d(inputs, weights, self.strides, self.padding) + self.bias
        return self.activation(pre_act)
