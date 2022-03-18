'''
Module containing the different models evaluated by experiments.
'''

from typing import List

from lowrank import LRConv2D, LRDense
from tensorflow.keras.layers import (AveragePooling2D, Conv2D, Dense, Flatten,
                                     InputLayer, MaxPool2D)
from tensorflow.keras.models import Sequential


def get_model(input_shape: List[int], num_classes: int, rank: int):
    '''
    :param input_shape: list of integers with input shape
    :param num_classes: number of classes of output
    :rank: initial rank for LRDense layer used in 2nd last layer
    :returns: Model with 4 convolutions layers, followed by a LRDense layer (rank passed as param) and a standard dense layer
    '''
    return Sequential(
        [
            InputLayer(input_shape=input_shape),
            Conv2D(64, 3, activation="relu", padding="same"),
            MaxPool2D(),
            Conv2D(128, 3, activation="relu", padding="same"),
            MaxPool2D(),
            Conv2D(256, 3, activation="relu", padding="same"),
            MaxPool2D(),
            Conv2D(512, 3, activation="relu", padding="same"),
            AveragePooling2D(),
            Flatten(),
            LRDense(256, rank=rank, activation="relu"),
            Dense(num_classes, activation="softmax"),
        ]
    )


def get_lr_conv_model(input_shape: List[int], num_classes: int, rank: int):
    '''
    :param input_shape: list of integers with input shape
    :param num_classes: number of classes of output
    :rank: initial rank for LRDense layer used in 2nd last layer
    :returns: Model with 4 LRConv layers with rank initialized to 32, followed by a LRDense layer (rank passed as param) and a standard dense layer
    '''
    return Sequential(
        [
            InputLayer(input_shape=input_shape),
            LRConv2D(64, 3, rank=32, activation="relu", padding="same"),
            MaxPool2D(),
            LRConv2D(128, 3, rank=32, activation="relu", padding="same"),
            MaxPool2D(),
            LRConv2D(256, 3, rank=32, activation="relu", padding="same"),
            MaxPool2D(),
            LRConv2D(512, 3, rank=32, activation="relu", padding="same"),
            AveragePooling2D(),
            Flatten(),
            LRDense(256, rank=rank, activation="relu"),
            Dense(num_classes, activation="softmax"),
        ]
    )

def get_unoptimized_lr__model(input_shape: List[int], num_classes: int):
    '''
    :param input_shape: list of integers with input shape
    :param num_classes: number of classes of output
    :rank: initial rank for LRDense layer used in 2nd last layer
    :returns: Model with 4 LRConv layers with no initial rank constraint, followed by a LRDense layer with no initial rank constraint and a standard dense layer
    '''
    return Sequential(
        [
            InputLayer(input_shape=input_shape),
            LRConv2D(64, 3, rank=-1, activation="relu", padding="same"),
            MaxPool2D(),
            LRConv2D(128, 3, rank=-1, activation="relu", padding="same"),
            MaxPool2D(),
            LRConv2D(256, 3, rank=-1, activation="relu", padding="same"),
            MaxPool2D(),
            LRConv2D(512, 3, rank=-1, activation="relu", padding="same"),
            AveragePooling2D(),
            Flatten(),
            LRDense(256, rank=-1, activation="relu"),
            Dense(num_classes, activation="softmax"),
        ]
    )

def get_vary_conv_rank_model(
    input_shape: List[int], num_classes: int, initial_ranks: List[int]
):
    '''
    :param input_shape: list of integers with input shape
    :param num_classes: number of classes of output
    :initial_ranks: list specifying initial ranks for each LRLayer
    :returns: Model with 4 LRConv layers with no initial rank constraint, followed by a LRDense layer with no initial rank constraint and a standard dense layer
    '''
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
