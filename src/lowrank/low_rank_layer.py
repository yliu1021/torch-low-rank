from copy import deepcopy
from typing import Optional

import torch
from torch import TensorType, nn, tensor
from torch.nn import functional as F


class LowRankLayer(nn.Module):
    def __init__(self, layer: nn.Module):
        super().__init__()

        # Copy over layer config
        self.layer_type = None
        if isinstance(layer, nn.Conv2d):
            self.in_channels = layer.in_channels
            self.out_channels = layer.out_channels
            self.kernel_size = layer.kernel_size
            self.stride = layer.stride
            self.padding = layer.padding
            self.dilation = layer.dilation
            self.groups = layer.groups
            self.padding_mode = layer.padding_mode
            self.layer_type = nn.Conv2d
        elif isinstance(layer, nn.Linear):
            self.in_features = layer.in_features
            self.out_features = layer.out_features
            self.layer_type = nn.Linear
        else:
            raise ValueError("Unsupported layer provided to LowRankLayer constructor.")

        # Copy over weights
        self.original_weights_shape = deepcopy(layer.weight.shape)
        if len(self.original_weights_shape) > 2:
            self.kernel_w = torch.reshape(
                layer.weight,
                (
                    self.in_channels * self.kernel_size[0] * self.kernel_size[1],
                    self.out_channels,
                ),
            )
        else:
            self.kernel_w = layer.weight
        self.bias = nn.Parameter(layer.bias)

        # Initialize other member variables
        self.kernel_u = None
        self.kernel_v = None
        self.kernel_uv = (self.kernel_u, self.kernel_v)
        self.eff_weights = None
        self._mask = None
        self._additional_mask = None

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

    def max_rank(self) -> int:
        return min(self.kernel_w.shape)

    @property
    def mask(self) -> Optional[TensorType]:
        return self._mask

    @mask.setter
    def mask(self, new_mask: Optional[TensorType]):
        """
        Sets a new mask, the shape of which determines masking mode. Setting to None reverts to
        full rank mode
        :param new_mask:
        :return:
        """
        if new_mask is None:
            self._mask = None
        elif len(new_mask.shape) == 1 and not self.svd_masking_mode:
            u, s, v = torch.linalg.svd(self.kernel_w, full_matrices=False)
            assert new_mask.shape == s.shape, "Invalid shape for mask"
            s_sqrt = torch.diag(torch.sqrt(s))
            self.kernel_u = torch.matmul(u, s_sqrt)
            self.kernel_v = torch.matmul(s_sqrt, v)
        elif len(self._mask.shape) == 2:
            assert new_mask.shape == self.kernel_w.shape, "Invalid shape for mask"

        # Reset additional mask if base mask changed
        self._additional_mask = None 

        self._mask = new_mask

        self.recompute_eff_weights()
        
    @property
    def additional_mask(self) -> Optional[TensorType]:
        """ Additional mask property, used to set mask while computing scores.
        New property needed to allow preserving current mask to enable iterative pruning.
        """
        return self._additional_mask

    @additional_mask.setter
    def additional_mask(self, new_additional_mask: Optional[TensorType]):
        """
        Additional Mask Setter -> Validates that a mask already exists and 
        that the new additional mask is of the same shape as the current mask
        """
        if new_additional_mask is None:
            self._additional_mask = None
        elif self._mask is None:
            raise ValueError("Cannot set additional mask with self.mask = None")
        elif new_additional_mask.shape != self._mask.shape:   
            raise ValueError("Additional mask must have same shape as mask")
        
        self._additional_mask = new_additional_mask

        self.recompute_eff_weights()

    def recompute_eff_weights(self):
        # Determine final mask 
        if self._additional_mask is not None:
            mask = torch.mul(self._mask, self._additional_mask)
        else:
            mask = self._mask

        # Compute effective weights
        if self.full_rank_mode:
            self.eff_weights = self.kernel_w
        elif self.weight_masking_mode:
            self.eff_weights = torch.mul(self.kernel_w, mask)
        else:
            self.eff_weights = self.kernel_u @ torch.diag(mask) @ self.kernel_v

    def forward(self, x):
        if self.eff_weights == None:
            self.recompute_eff_weights()

        # Do actual forward pass
        if self.layer_type is nn.Linear:
            return F.linear(input=x, weight=self.eff_weights, bias=self.bias)
        elif self.layer_type is nn.Conv2d:
            return F.conv2d(
                input=x,
                weight=torch.reshape(self.eff_weights, self.original_weights_shape),
                bias=self.bias,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups,
            )
        else:
            raise RuntimeError("Layer in bad state, layer type corrupted")


if __name__ == "__main__":
    test_input = tensor([1.0, 1.0, 1.0])

    """
    Low Rank Linear Test 
    """
    linear = nn.Linear(3, 5)
    lr_linear = LowRankLayer(linear)

    # Sanity check
    assert (linear.forward(test_input) == lr_linear.forward(test_input)).all()

    # Low Rank Test
    lr_linear.mask = tensor([1.0, 1.0, 0.0])

    lr_linear.forward(test_input)

    """
    Conv2D Test 
    """

    conv2d = nn.Conv2d(3, 3, (1, 1), 1)
    lr_conv2d = LowRankLayer(conv2d)

    test_input = tensor([[[1.0]], [[1.0]], [[1.0]]])

    # Sanity check
    assert (lr_conv2d.forward(test_input) == conv2d.forward(test_input)).all()

    # Low Rank Test
    lr_conv2d.mask = tensor([1.0, 1.0, 0.0])

    lr_conv2d.forward(test_input)
