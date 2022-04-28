from typing import Dict, Optional, Tuple, Union

import numpy as np
import tensorflow as tf
from tensorflow.keras import activations, regularizers
from tensorflow.keras.initializers import GlorotUniform
from tensorflow.keras.layers import Layer


class LowRankLayer(Layer):
    def __init__(
        self,
        activation: Optional[str] = None,
        weight_decay: float = 1e-4,
        **kwargs,
    ):
        """
        Creates a LRLayer that starts in full rank mode.
        :param activation: activation function to apply
        :param weight_decay: L2 weight decay
        """
        super().__init__(**kwargs)
        self.activation = activations.get(activation)
        self.weight_decay = weight_decay
        # start in full rank mode
        self._mask: Optional[tf.Variable] = None
        self.freeze_v = False
        # properties to be created after layer is built
        self.num_inputs: Optional[int] = None
        self.num_outputs: Optional[int] = None
        self.kernel_w: Optional[tf.Variable] = None
        self.kernel_uv: Optional[Tuple[tf.Variable, tf.Variable]] = None
        self.bias: Optional[tf.Variable] = None

    @property
    def max_rank(self) -> int:
        """
        The maximum possible rank this layer can have
        :return: the max rank
        """
        return min(self.num_outputs, self.num_inputs)

    @property
    def full_rank_mode(self) -> bool:
        """
        LRLayers can express the kernel using a single W matrix or UV matrix tuple. Full rank mode
        refers to when the layer is using W (with no mask because sparse matrices are low rank)
        :return: True if using W matrix with no mask. False if using UV tuple (masked or not)
        """
        return self._mask is None

    @property
    def svd_masking_mode(self) -> bool:
        """
        Returns True iff we're masking in singular vectors
        :return: True iff masking singular vectors, False otherwise
        """
        return self._mask is not None and len(self._mask.shape) == 1

    @property
    def weight_masking_mode(self) -> bool:
        """
        Returns True iff we're masking weights
        :return: True iff masking weights, False otherwise
        """
        return self._mask is not None and len(self._mask.shape) != 1

    # we can assume that *exactly one* property to be true at all times:
    # 1. `full_rank_mode`,
    # 2. `svd_masking_mode`,
    # 3. `weight_masking_mode`

    @property
    def rank(self) -> int:
        """
        The rank of the layer if in low rank mode.
        :return: The number of unmasked singular vectors if in low rank mode. Returns max rank
        otherwise.
        """
        if self.full_rank_mode:
            return self.max_rank
        if self.svd_masking_mode:
            return int(sum(self._mask.numpy()))
        if self.weight_masking_mode:
            # TODO: it's debatable that a sparse matrix is not full rank technically
            return self.max_rank
        raise RuntimeError("Invalid layer state")

    @property
    def mask(self) -> Optional[tf.Variable]:
        return self._mask

    @mask.setter
    def mask(self, new_mask: Optional[Union[tf.Variable, np.ndarray]]):
        """
        Sets a new mask, the shape of which determines masking mode. Setting to None reverts to
        full rank mode
        :param new_mask:
        :return:
        """
        # optimization technique to avoid redundant eff_weight and SVD calculation
        if self.svd_masking_mode and len(new_mask.shape) == 1:
            self._mask.assign(new_mask, read_value=True)
            return
        if self.weight_masking_mode and len(new_mask.shape) == 2:
            self._mask.assign(new_mask, read_value=True)
            return
        eff_weights = self.eff_weight()
        if new_mask is None:
            self._mask = None
        else:
            self._mask = tf.Variable(new_mask, trainable=False, dtype=tf.float32)
        if self.full_rank_mode or self.weight_masking_mode:
            self.kernel_w.assign(eff_weights, read_value=True)
        elif self.svd_masking_mode:
            u, s, v = np.linalg.svd(eff_weights, full_matrices=False)
            s = np.diag(s)
            u = u @ (s**0.5)
            v = (s**0.5) @ v
            u_weights, v_weights = self.kernel_uv
            u_weights.assign(u, read_value=True)
            v_weights.assign(v, read_value=True)
        else:
            raise RuntimeError("Invalid layer state")

    def eff_weight(self) -> tf.Variable:
        """
        Return the effective weights of the layer (with masking).
        If the layer is in SVD form, return U @ V
        :return: effective weights
        """
        if not self.built:
            raise RuntimeError(
                "Layer must be built first before calling effective weights"
            )
        if self.full_rank_mode:
            return self.kernel_w
        if self.svd_masking_mode:
            u, v = self.kernel_uv
            return u @ tf.linalg.diag(self._mask) @ v
        if self.weight_masking_mode:
            return self.kernel_w * self._mask
        raise RuntimeError("Invalid layer state")

    @property
    def trainable_weights(self):
        if self.full_rank_mode or self.weight_masking_mode:
            weights = [self.kernel_w]
        else:
            assert self.svd_masking_mode, "Must be in svd masking mode"
            u, v = self.kernel_uv
            weights = [u]
            if not self.freeze_v:
                weights.append(v)
        if self.bias is not None:
            weights.append(self.bias)
        return weights

    def get_config(self):
        config = super().get_config()
        config.update(
            {"activation": self.activation, "weight_decay": self.weight_decay}
        )
        return config

    def _build_weights(self):
        """
        Allocate tensorflow weights after layer is built
        """
        if self.kernel_w is None:
            self.kernel_w = self.add_weight(
                name="kernel_w",
                shape=(self.num_inputs, self.num_outputs),
                initializer=GlorotUniform(),
                regularizer=regularizers.l2(self.weight_decay),
            )
        if self.kernel_uv is None:
            u = self.add_weight(
                name=f"kernel_u",
                shape=(self.num_inputs, self.max_rank),
                initializer=GlorotUniform(),
            )
            v = self.add_weight(
                name=f"kernel_v",
                shape=(self.max_rank, self.num_outputs),
                initializer=GlorotUniform(),
            )
            self.add_loss(self.weight_decay * tf.norm(u @ v))
            self.kernel_uv = (u, v)
