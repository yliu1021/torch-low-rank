from copy import deepcopy
from typing import Optional
import torch 
from torch import TensorType, nn 
from torch.nn import functional as F

class LowRankLayer(nn.Module):
    def __init__(self, layer: nn.Module):

        # Copy over layer config
        self.layer_type = None
        # Conv2D 
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
        # Linear2D    
        elif isinstance(layer, nn.Linear):
            self.in_features = layer.in_features
            self.out_features = layer.out_features
            self.layer_type = nn.Linear
        else:
            raise ValueError("Unsupported layer provided to LowRankLayer constructor.")
            
        # Copy over weights
        self.original_weights_shape = deepcopy(layer.weight.shape)
        if len(self.original_weights_shape) > 2:
            self.w = torch.reshape(layer.weight, (self.in_channels * self.kernel_size[0] * self.kernel_size[1], self.out_channels))
        else:
            self.w = layer.weight
        self.bias = nn.Parameter(layer.bias)

        # Initialize other member variables
        self.u = None 
        self.v = None 
        self._mask = None 
    
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

    @property
    def mask(self) -> Optional[nn.Parameter]:
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
        else:
            self._mask = nn.Parameter(new_mask, requires_grad=False)

        if self.svd_masking_mode:
            u, s, v = torch.svd(self.w, some=True)
            s_sqrt = torch.diag(torch.sqrt(s))
            self.u = torch.matmul(u, s_sqrt)
            self.v = torch.matmul(s_sqrt, v)

    def forward(self, x):
        # Compute effective weights
        if self.full_rank_mode:
            eff_weights = self.w
        elif self.weight_masking_mode:
            eff_weights = torch.mul(self.w, self.mask)
        else: 
            eff_weights = torch.matmul(self.u, torch.matmul(self._mask, self.v))
        
        # Do actual forward pass
        if self.layer_type is nn.Linear:
            return F.linear(
                input=x,
                weight=eff_weights,
                bias=self.bias
            )
        elif self.layer_type is nn.Conv2d:
            return F.conv2d(
                input=x,
                weight=torch.reshape(eff_weights, self.original_weights_shape),
                bias=self.bias,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups
            )
        else:
            raise RuntimeError("Layer in bad state, layer type corrupted")


        