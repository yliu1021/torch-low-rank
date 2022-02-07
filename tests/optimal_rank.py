import enum
from typing import Tuple, Union

import lowrank
import numpy as np
from lowrank import low_rank_layer
from tensorflow.keras import models


class OptimizationMethod(enum.Enum):
    BACK_TO_FRONT = enum.auto()
    FRONT_TO_BACK = enum.auto()


def optimize_ranks(
    model: models.Model,
    data: Tuple[np.ndarray, np.ndarray],
    layer_order: OptimizationMethod,
):
    """
    Optimizes the ranks of a low rank model by monitoring training loss before and after
    compression at various epochs
    :param model: the model to optimize (this model's layer ranks will be changed inplace)
    :param data: the training data to use for optimization
    :param layer_order: the order to optimize the ranks
    `OptimizationMethod` to optimize multiple layers
    """
    num_layers = len(model.layers)
    if layer_order is OptimizationMethod.BACK_TO_FRONT:
        layer_inds = reversed(range(num_layers))
    elif layer_order is OptimizationMethod.FRONT_TO_BACK:
        layer_inds = range(num_layers)
    else:
        raise NotImplementedError(f"Layer order: {layer_order} not supported")
    for i in layer_inds:
        if isinstance(model.layers[i], low_rank_layer.LowRankLayer):
            optimize_rank_of_layer(model, data, i)


def optimize_rank_of_layer(
    model: models.Model,
    data: Tuple[np.ndarray, np.ndarray],
    layer_ind: int,
    batch_size: int = 64,
    epochs: int = 3
):
    """
    Optimizes the rank of just 1 layer specified by `layer_ind`
    :param model: the model that the layer belongs in
    :param data: the data to use for training
    :param layer_ind: the index of the layer to optimize
    :param batch_size: the batch size for training
    :param epochs: the number of epochs to use for training
    """
    layer = model.layers[layer_ind]
    if not isinstance(layer, low_rank_layer.LowRankLayer):
        raise ValueError(f"Layer must be a low rank layer. Got a {type(layer)} instead")
    layer.set_rank(-1)  # start full rank
    model._reset_compile_cache()
    initial_weights = model.get_weights()
    x, y = data

    loss = model.fit(x, y, batch_size=batch_size)
    initial_loss = loss.history["loss"][0]
    epoch_start_weights = model.get_weights()
    for _ in range(epochs - 1):
        loss = model.fit(x, y, batch_size=batch_size)
    final_loss = loss.history["loss"][0]
    epoch_final_weights = model.get_weights()
    # TODO: compute the losses at each epoch and restore weights to compare them when they're
    # TODO: compressed


def _compute_new_rank_acc(
    model: models.Model,
    layer: low_rank_layer.LowRankLayer,
    new_rank: int,
    data: Tuple[np.ndarray, np.ndarray],
):
    curr_rank = layer.rank
    curr_weights = layer.weights
    layer.set_rank(new_rank)
    model._reset_compile_cache()
    x, y = data
    losses = model.evaluate(x, y)
    loss = losses["loss"]
    layer.set_rank(curr_rank)
    layer.set_weights(curr_weights)
    model._reset_compile_cache()
    return loss
