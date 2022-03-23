from typing import List

from tensorflow.keras.layers import (
    AveragePooling2D,
    Dense,
    Flatten,
    InputLayer,
    MaxPool2D,
)
from tensorflow.keras.models import Sequential
import tensorflow as tf 

from lowrank import LRConv2D, LRDense

class LowRankModel(tf.keras.Model):
    def __init__(self, input_shape: List[int], num_classes: int, initial_ranks: List[int]) -> None:
        """
        :param input_shape: list of integers with input shape
        :param num_classes: number of classes of output
        :param initial_ranks: list specifying initial ranks for each LRLayer
        :returns: Model with 4 LRConv layers with no initial rank constraint,
        followed by a LRDense layer with no initial rank constraint and a standard dense layer
        """
        super().__init__()
        if len(initial_ranks) != 5:
            raise ValueError(
            f"Must specify 5 initial ranks. Given {len(initial_ranks)} instead"
        )

        self.input_shape = input_shape
        self.num_classes = num_classes
        self.initial_ranks = initial_ranks
        
        self.layers = Sequential(
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

    def call(self, x):
        return self.layers(x)