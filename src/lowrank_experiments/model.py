"""
Module containing models
"""

from typing import List

from tensorflow.keras.applications import vgg16
from tensorflow.keras.layers import (
    AveragePooling2D,
    BatchNormalization,
    Dense,
    Dropout,
    Flatten,
    InputLayer,
    MaxPool2D
)
from tensorflow.keras.models import Sequential

from lowrank import LRConv2D, LRDense


def get_lr_model(input_shape: List[int], num_classes: int):
    """
    Returns a Low Rank Compatible Model

    :param input_shape: list of integers with input shape
    :param num_classes: number of classes of output
    :param initial_ranks: list specifying initial ranks for each LRLayer (if none, each layer starts out full rank)

    :returns: Model with 4 LRConv layers with no initial rank constraint, followed by a LRDense layer with no initial rank constraint and a standard dense layer
    """
    return Sequential(
        [
            InputLayer(input_shape=input_shape),
            LRConv2D(64, 3, activation="relu", padding="same"),
            MaxPool2D(),
            LRConv2D(128, 3, activation="relu", padding="same"),
            MaxPool2D(),
            LRConv2D(256, 3, activation="relu", padding="same"),
            MaxPool2D(),
            LRConv2D(512, 3, activation="relu", padding="same"),
            AveragePooling2D(),
            Flatten(),
            LRDense(256, activation="relu"),
            Dense(num_classes, activation="softmax"),
        ]
    )


def get_lr_vgg19(
    input_shape: List[int],
    num_classes: int,
    weight_decay: float = 0,
):
    layers = []
    for num_filters_blocks in [[64] * 3, [128] * 2, [256] * 4, [512] * 4, [512] * 4]:
        for num_filters in num_filters_blocks:
            layers.extend(
                [
                    LRConv2D(
                        num_filters,
                        3,
                        activation="relu",
                        padding="same",
                        weight_decay=weight_decay,
                    ),
                    BatchNormalization(),
                ]
            )
        layers.append(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    layers.extend(
        [
            Flatten(),
            LRDense(
                4096,
                activation="relu",
                weight_decay=weight_decay,
            ),
            BatchNormalization(),
            LRDense(
                4096,
                activation="relu",
                weight_decay=weight_decay,
            ),
            BatchNormalization(),
            LRDense(num_classes, activation="softmax", weight_decay=weight_decay),
        ]
    )
    return Sequential([InputLayer(input_shape=input_shape), *layers])


def get_lr_vgg16(
    input_shape: List[int],
    num_classes: int,
    weight_decay: float = 0,
):
    """
    Returns VGG16 Low Rank Compatible

    :param input_shape: list of integers with input shape
    :param num_classes: number of classes of output
    :param weight_decay: l2 reg term
    :returns: VGG16
    """
    return Sequential(
        [
            InputLayer(input_shape=input_shape),
            LRConv2D(
                64,
                3,
                activation="relu",
                padding="same",
                weight_decay=weight_decay,
            ),
            BatchNormalization(),
            LRConv2D(
                64,
                3,
                activation="relu",
                padding="same",
                weight_decay=weight_decay,
            ),
            BatchNormalization(),
            MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
            LRConv2D(
                128,
                3,
                activation="relu",
                padding="same",
                weight_decay=weight_decay,
            ),
            BatchNormalization(),
            LRConv2D(
                128,
                3,
                activation="relu",
                padding="same",
                weight_decay=weight_decay,
            ),
            BatchNormalization(),
            MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
            LRConv2D(
                256,
                3,
                activation="relu",
                padding="same",
                weight_decay=weight_decay,
            ),
            BatchNormalization(),
            LRConv2D(
                256,
                3,
                activation="relu",
                padding="same",
                weight_decay=weight_decay,
            ),
            BatchNormalization(),
            LRConv2D(
                256,
                3,
                activation="relu",
                padding="same",
                weight_decay=weight_decay,
            ),
            BatchNormalization(),
            MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
            LRConv2D(
                512,
                3,
                activation="relu",
                padding="same",
                weight_decay=weight_decay,
            ),
            BatchNormalization(),
            LRConv2D(
                512,
                3,
                activation="relu",
                padding="same",
                weight_decay=weight_decay,
            ),
            BatchNormalization(),
            LRConv2D(
                512,
                3,
                activation="relu",
                padding="same",
                weight_decay=weight_decay,
            ),
            BatchNormalization(),
            MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
            LRConv2D(
                512,
                3,
                activation="relu",
                padding="same",
                weight_decay=weight_decay,
            ),
            BatchNormalization(),
            LRConv2D(
                512,
                3,
                activation="relu",
                padding="same",
                weight_decay=weight_decay,
            ),
            BatchNormalization(),
            LRConv2D(
                512,
                3,
                activation="relu",
                padding="same",
                weight_decay=weight_decay,
            ),
            BatchNormalization(),
            MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
            Flatten(),
            LRDense(
                4096,
                activation="relu",
                weight_decay=weight_decay,
            ),
            Dropout(0.5),
            BatchNormalization(),
            LRDense(
                4096,
                activation="relu",
                weight_decay=weight_decay,
            ),
            Dropout(0.5),
            BatchNormalization(),
            LRDense(num_classes, activation="softmax", weight_decay=weight_decay),
        ]
    )


def get_lr_vgg11(
    input_shape: List[int],
    num_classes: int,
    weight_decay: float = 0,
):
    """
    Returns VGG11 Low Rank Compatible

    :param input_shape: list of integers with input shape
    :param num_classes: number of classes of output
    :param weight_decay: l2 reg term
    :returns: VGG11
    """
    return Sequential(
        [
            InputLayer(input_shape=input_shape),
            LRConv2D(
                64,
                3,
                activation="relu",
                padding="same",
                weight_decay=weight_decay,
            ),
            BatchNormalization(),
            MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
            LRConv2D(
                128,
                3,
                activation="relu",
                padding="same",
                weight_decay=weight_decay,
            ),
            BatchNormalization(),
            MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
            LRConv2D(
                256,
                3,
                activation="relu",
                padding="same",
                weight_decay=weight_decay,
            ),
            BatchNormalization(),
            LRConv2D(
                256,
                3,
                activation="relu",
                padding="same",
                weight_decay=weight_decay,
            ),
            BatchNormalization(),
            MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
            LRConv2D(
                512,
                3,
                activation="relu",
                padding="same",
                weight_decay=weight_decay,
            ),
            BatchNormalization(),
            LRConv2D(
                512,
                3,
                activation="relu",
                padding="same",
                weight_decay=weight_decay,
            ),
            BatchNormalization(),
            MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
            LRConv2D(
                512,
                3,
                activation="relu",
                padding="same",
                weight_decay=weight_decay,
            ),
            BatchNormalization(),
            LRConv2D(
                512,
                3,
                activation="relu",
                padding="same",
                weight_decay=weight_decay,
            ),
            BatchNormalization(),
            MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
            Flatten(),
            LRDense(
                4096,
                activation="relu",
                weight_decay=weight_decay,
            ),
            Dropout(0.5),
            BatchNormalization(),
            LRDense(
                4096,
                activation="relu",
                weight_decay=weight_decay,
            ),
            Dropout(0.5),
            BatchNormalization(),
            Dense(num_classes, activation="softmax"),
        ]
    )


def get_vgg16(input_shape: List[int], num_classes: int):
    return vgg16.VGG16(
        include_top=True,
        weights=None,
        input_shape=input_shape,
        pooling="max",
        classes=num_classes,
    )


def get_model(
    model_name: str,
    input_shape: List[int],
    num_classes: int,
    weight_decay: float = 0,
):
    if model_name == "default":
        model = get_lr_model(input_shape, num_classes=num_classes)
    elif model_name == "vgg11":
        model = get_lr_vgg11(
            input_shape,
            num_classes=num_classes,
            weight_decay=weight_decay,
        )
    elif model_name == "vgg16":
        model = get_lr_vgg16(
            input_shape,
            num_classes=num_classes,
            weight_decay=weight_decay,
        )
    elif model_name == "vgg16_normal":
        model = get_vgg16(input_shape, num_classes=num_classes)
    elif model_name == "vgg19":
        model = get_lr_vgg19(
            input_shape,
            num_classes=num_classes,
            weight_decay=weight_decay,
        )
    else:
        raise NotImplementedError(model_name + " is not supported currently.")
    return model
