"""
Module containing models
"""

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


def get_lr_model(
    input_shape: List[int], num_classes: int, initial_ranks: List[int] = None
):
    """
    Returns a Low Rank Compatible Model

    :param input_shape: list of integers with input shape
    :param num_classes: number of classes of output
    :param initial_ranks: list specifying initial ranks for each LRLayer (if none, each layer starts out full rank)

    :returns: Model with 4 LRConv layers with no initial rank constraint, followed by a LRDense layer with no initial rank constraint and a standard dense layer
    """
    if initial_ranks == None:
        initial_ranks = [-1] * 5

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


def get_lr_vgg16(
    input_shape: List[int], num_classes: int, initial_ranks: List[int] = None
):
    """
    Returns VGG16 Low Rank Compatible

    :param input_shape: list of integers with input shape
    :param num_classes: number of classes of output
    :param initial_ranks: list specifying initial ranks for each LRLayer (if none, each layer starts out full rank)

    :returns: VGG16
    """
    if initial_ranks == None:
        initial_ranks = [-1] * 16

    if len(initial_ranks) != 16:
        raise ValueError(
            f"Must specify 16 initial ranks. Given {len(initial_ranks)} instead"
        )

    return Sequential(
        [
            InputLayer(input_shape=input_shape),
            LRConv2D(64, 3, rank=initial_ranks[0], activation="relu", padding="same"),
            LRConv2D(64, 3, rank=initial_ranks[1], activation="relu", padding="same"),
            MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
            LRConv2D(128, 3, rank=initial_ranks[2], activation="relu", padding="same"),
            LRConv2D(128, 3, rank=initial_ranks[3], activation="relu", padding="same"),
            MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
            LRConv2D(256, 3, rank=initial_ranks[4], activation="relu", padding="same"),
            LRConv2D(256, 3, rank=initial_ranks[5], activation="relu", padding="same"),
            LRConv2D(256, 3, rank=initial_ranks[6], activation="relu", padding="same"),
            MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
            LRConv2D(512, 3, rank=initial_ranks[7], activation="relu", padding="same"),
            LRConv2D(512, 3, rank=initial_ranks[8], activation="relu", padding="same"),
            LRConv2D(512, 3, rank=initial_ranks[9], activation="relu", padding="same"),
            MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
            LRConv2D(512, 3, rank=initial_ranks[10], activation="relu", padding="same"),
            LRConv2D(512, 3, rank=initial_ranks[11], activation="relu", padding="same"),
            LRConv2D(512, 3, rank=initial_ranks[12], activation="relu", padding="same"),
            MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
            Flatten(),
            LRDense(4096, rank=initial_ranks[13], activation="relu"),
            LRDense(4096, rank=initial_ranks[14], activation="relu"),
            LRDense(1000, rank=initial_ranks[15], activation="relu"),
            Dense(num_classes, activation="softmax"),
        ]
    )


def get_lr_vgg11(
    input_shape: List[int], num_classes: int, initial_ranks: List[int] = None
):
    """
    Returns VGG11 Low Rank Compatible

    :param input_shape: list of integers with input shape
    :param num_classes: number of classes of output
    :param initial_ranks: list specifying initial ranks for each LRLayer (if none, each layer starts out full rank)

    :returns: VGG11
    """
    if initial_ranks == None:
        initial_ranks = [-1] * 11

    if len(initial_ranks) != 11:
        raise ValueError(
            f"Must specify 11 initial ranks. Given {len(initial_ranks)} instead"
        )

    return Sequential(
        [
            InputLayer(input_shape=input_shape),
            LRConv2D(64, 3, rank=initial_ranks[0], activation="relu", padding="same"),
            MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
            LRConv2D(128, 3, rank=initial_ranks[1], activation="relu", padding="same"),
            MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
            LRConv2D(256, 3, rank=initial_ranks[2], activation="relu", padding="same"),
            LRConv2D(256, 3, rank=initial_ranks[3], activation="relu", padding="same"),
            MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
            LRConv2D(512, 3, rank=initial_ranks[4], activation="relu", padding="same"),
            LRConv2D(512, 3, rank=initial_ranks[5], activation="relu", padding="same"),
            MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
            LRConv2D(512, 3, rank=initial_ranks[6], activation="relu", padding="same"),
            LRConv2D(512, 3, rank=initial_ranks[7], activation="relu", padding="same"),
            MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
            Flatten(),
            LRDense(4096, rank=initial_ranks[8], activation="relu"),
            LRDense(4096, rank=initial_ranks[9], activation="relu"),
            LRDense(1000, rank=initial_ranks[10], activation="relu"),
            Dense(num_classes, activation="softmax"),
        ]
    )
