from typing import List

from tensorflow.keras.layers import (
    AveragePooling2D,
    Dense,
    Flatten,
    InputLayer,
    MaxPool2D,
)
from tensorflow.keras.models import Sequential

from lowrank import LRConv2D, LRDense


def get_lr_model(input_shape: List[int], num_classes: int, initial_ranks: List[int]):
    """
    :param input_shape: list of integers with input shape
    :param num_classes: number of classes of output
    :param initial_ranks: list specifying initial ranks for each LRLayer
    :returns: Model with 4 LRConv layers with no initial rank constraint,
    followed by a LRDense layer with no initial rank constraint and a standard dense layer
    """
    if len(initial_ranks) != 5:
        raise ValueError(
            f"Must specify 5 initial ranks. Given {len(initial_ranks)} instead"
        )
    return Sequential(
        [
            InputLayer(input_shape=input_shape),
            LRConv2D(64, 3, rank=initial_ranks[0], activation="relu", padding="same"),
            MaxPool2D(),
            LRConv2D(128, 3, rank=initial_ranks[1], activation="relu", padding="same"),
            MaxPool2D(),
            LRConv2D(256, 3, rank=initial_ranks[2], activation="relu", padding="same"),
            MaxPool2D(),
            LRConv2D(512, 3, rank=initial_ranks[3], activation="relu", padding="same"),
            AveragePooling2D(),
            Flatten(),
            LRDense(256, rank=initial_ranks[4], activation="relu"),
            Dense(num_classes, activation="softmax"),
        ]
    )
