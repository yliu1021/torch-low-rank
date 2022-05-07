from torch import nn
from torchvision import models

from lowrank.low_rank_layer import LowRankLayer

pytorch = models


class BasicNet(nn.Module):
    def __init__(self):
        super(BasicNet, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(32 * 32 * 3, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


def convert_module_to_lr(module: nn.Module):
    """
    Recursively modifies a module in place to replace instances of conv2d and linear modules into
    low rank alternatives
    :param module: the module to convert
    :return: the converted module
    """
    for attr_name in dir(module):
        attr = getattr(module, attr_name)
        if isinstance(attr, (nn.Linear, nn.Conv2d)):
            setattr(module, attr_name, LowRankLayer(attr))
        elif isinstance(attr, nn.Module):
            convert_module_to_lr(attr)
        elif isinstance(attr, list):
            attr = [convert_module_to_lr(a) for a in attr]
            setattr(module, attr_name, attr)
    
    return module