"""
Module for loading and pre-processing data.
"""

from typing import Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import cifar10, cifar100

Dataset = Tuple[np.ndarray, np.ndarray]


def _normalize_x(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32) / 255.0
    x -= x.mean(axis=(0,1,2))
    x /= x.std(axis=(0,1,2))
    return x


def _one_hot_y(y: np.ndarray) -> np.ndarray:
    num_classes = np.max(y) + 1
    y = np.squeeze(y)
    return np.array(tf.one_hot(y, depth=num_classes))


def load_data(dataset: str, fast: bool = False) -> Tuple[Dataset, Dataset]:
    """
    Loads a dataset and adds noisy labels to `noise` percent of the training labels
    :param dataset: the dataset to load: "cifar10" or "cifar100"
    :return: a tuple of Datasets for training and testing
    """
    if dataset == "cifar10":
        load = cifar10.load_data
    elif dataset == "cifar100":
        load = cifar100.load_data
    else:
        raise ValueError(f"Invalid dataset: {dataset}")
    (x_train, y_train), (x_test, y_test) = load()
    x_train = _normalize_x(x_train)
    x_test = _normalize_x(x_test)
    y_train = _one_hot_y(y_train)
    y_test = _one_hot_y(y_test)

    if fast:
        x_train = x_train[:64, :, :, :]
        y_train = y_train[:64, :]
        x_test = x_test[:64, :, :, :]
        y_test = y_test[:64, :]

    return (x_train, y_train), (x_test, y_test)
