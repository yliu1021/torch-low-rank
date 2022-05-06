from torch import nn
from torchvision import models

import lowrank

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


def convert_module(module: nn.Module):
    """
    Recursively modifies a module in place to replace instances of conv2d and linear modules into
    low rank alternatives
    :param module: the module to convert
    :return:
    """
    for attr_name in dir(module):
        attr = getattr(module, attr_name)
        if isinstance(attr, (nn.Linear, nn.Conv2d)):
            setattr(module, attr_name, lowrank.LowRankLayer(attr))
        elif isinstance(attr, nn.Module):
            convert_module(attr)
        elif isinstance(attr, list):
            attr = [convert_module(a) for a in attr]
            setattr(module, attr_name, attr)
