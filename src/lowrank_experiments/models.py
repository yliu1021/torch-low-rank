from pyexpat import model
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


def convert_model_to_lr(model: nn.Module):
    """
    Recursively modifies a module in place to replace instances of conv2d and linear modules into
    low rank alternatives
    :param module: the module to convert
    :return:
    """
    if isinstance(model, (nn.Linear, nn.Conv2d)):
        return LowRankLayer(model)

    for attr_name in dir(model):
        attr = getattr(model, attr_name)
        if isinstance(attr, nn.ModuleList) or isinstance(attr, nn.Sequential):
            for i in range(len(attr)):
                attr[i] = convert_model_to_lr(attr[i])
        elif isinstance(attr, nn.ModuleDict):
            for key in attr.keys():
                attr[key] = convert_model_to_lr(attr[key])
        elif isinstance(attr, nn.Module):
            attr = convert_model_to_lr(attr)
        else:
            continue
        setattr(model, attr_name, attr)

    assert not any([isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear) for layer in model.modules()]), "Convert to lr model failed."
    
    return model