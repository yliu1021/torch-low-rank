import argparse
import enum
from pprint import pprint
from typing import Tuple

import numpy as np
import tensorflow as tf
from lowrank import low_rank_layer
from tensorflow.keras import losses, metrics, models, optimizers
from tensorflow.python.framework.ops import Tensor
from tensorflow.python.training.tracking import base

import data
from model import get_unoptimized_lr__model


class OptimizationMethod(enum.Enum):
    BACK_TO_FRONT = enum.auto()
    FRONT_TO_BACK = enum.auto()

def optimize_ranks(
    model: models.Model,
    model_copy: models.Model,
    data: Tuple[np.ndarray, np.ndarray],
    layer_order: OptimizationMethod = OptimizationMethod.BACK_TO_FRONT,
    epochs: int = 3,
):
    """
    Optimizes the ranks of a low rank model by monitoring training loss before and after
    compression at various epochs
    :param model: the model to optimize (this model's layer ranks will be changed inplace)
    :param data: the training data to use for optimization
    :param layer_order: `OptimizationMethod` the order to optimize the ranks 
    :param epochs: max number of epochs to train for while optimizing ranks
    """
    num_layers = len(model.layers)
    if layer_order is OptimizationMethod.BACK_TO_FRONT:
        layer_inds = reversed(range(num_layers))
    elif layer_order is OptimizationMethod.FRONT_TO_BACK:
        layer_inds = range(num_layers)
    else:
        raise NotImplementedError(f"Layer order: {layer_order} not supported")
    
    '''
    Train Model
    ''' 
    print("Training Model")
    x, y = data
    model_losses = []
    model.compile(
            optimizer=optimizers.Adam(0.0001),
            loss=losses.CategoricalCrossentropy(),
            metrics=[metrics.CategoricalAccuracy()],
            run_eagerly=True,
        )

    # Train for 1 epoch and save copy of model
    loss = model.fit(x, y, batch_size=64)
    model_losses.extend(loss.history["loss"])
    # TODO: Figure out how to clone model correctly
    # initial_model = models.clone_model(model) - this fails with error
    initial_model = model_copy
    initial_model.compile(
        optimizer=optimizers.Adam(0.0001),
        loss=losses.CategoricalCrossentropy(),
        metrics=[metrics.CategoricalAccuracy()],
        run_eagerly=True,
    )
    initial_model.set_weights(model.get_weights()) 
    print("Training For Initial Model (1 Epoch) Complete.")

    # Train for remaining epochs
    loss = model.fit(x, y, batch_size=64, epochs=epochs-1)
    model_losses.extend(loss.history["loss"])

    print("Training Complete.")

    baseline_initial_loss = evaluate_model_loss(initial_model, data)
    baseline_final_loss = evaluate_model_loss(model, data)
    
    '''
    Iterate through layers to determine optimal ranks
    '''
    optimal_ranks = {}
    for i in layer_inds:
        if not isinstance(model.layers[i], low_rank_layer.LowRankLayer):
            continue
        
        print("Determining Optimal Rank for Layer " + str(i))

        optimal_rank = optimize_rank_of_layer(
            initial_model=initial_model,
            final_model=model,
            data=data,
            layer_ind=i,
            baseline_initial_loss=baseline_initial_loss,
            baseline_final_loss=baseline_final_loss)
        optimal_ranks[i] = optimal_rank
        
        print("Optimal Rank for Layer " + str(i) + " = " + str(optimal_rank))

        baseline_initial_loss = set_optimal_rank(initial_model, optimal_rank, i, data)
        baseline_final_loss = set_optimal_rank(model, optimal_rank, i, data)
    
    print("Optimal Ranks: ")
    pprint(optimal_ranks)

    return model

def set_optimal_rank(
    model: models.Model,
    optimal_rank: int,
    layer_ind: int,
    data: Tuple[np.ndarray, np.ndarray]
):
    model.layers[layer_ind].set_rank(optimal_rank)
    return evaluate_model_loss(model, data)

def evaluate_model_loss(
    model: models.Model,
    data: Tuple[np.ndarray, np.ndarray]
):
    model._reset_compile_cache()
    x, y = data
    losses = model.evaluate(x, y, batch_size=64)
    return losses[0]

def optimize_rank_of_layer(
    initial_model: models.Model,
    final_model: models.Model,
    data: Tuple[np.ndarray, np.ndarray],
    layer_ind: int,
    baseline_initial_loss: float,
    baseline_final_loss: float
):
    """
    Optimizes the rank of just 1 layer specified by `layer_ind`
    :param model: the model that the layer belongs in
    :param data: the data to use for training
    :param layer_ind: the index of the layer to optimize
    """
    if not isinstance(initial_model.layers[layer_ind], low_rank_layer.LowRankLayer):
        raise ValueError(f"Layer must be a low rank layer. Got a {type(initial_model.layers[layer_ind])} instead")
    
    print("Baseline Initial Loss", baseline_initial_loss)
    print("Baseline Final Loss", baseline_final_loss)

    max_insufficient_rank = 1
    min_sufficient_rank = max_rank(initial_model.layers[layer_ind])
    while(min_sufficient_rank > max_insufficient_rank + 1):
        print("Min Sufficient Rank", min_sufficient_rank)
        print("Max Insufficient Rank", max_insufficient_rank)
        new_rank = int((min_sufficient_rank + max_insufficient_rank) / 2)
        if is_sufficient_rank(new_rank, initial_model, final_model, data, layer_ind, baseline_initial_loss, baseline_final_loss):
            min_sufficient_rank = new_rank
        else:
            max_insufficient_rank = new_rank

    return min_sufficient_rank

def max_rank(lr_layer: low_rank_layer):
    shape = lr_layer.eff_weight().shape
    if not len(shape) == 2:
        raise ValueError(f"Layer's weights must be formatted as a 2D Tensor")
    return min(shape)

def is_sufficient_rank(
    new_rank: int,
    initial_model: models.Model,
    final_model: models.Model,
    data: Tuple[np.ndarray, np.ndarray],
    layer_ind: int, 
    baseline_initial_loss: float,
    baseline_final_loss: float
):
    lowrank_initial_loss = compute_new_rank_loss(initial_model, initial_model.layers[layer_ind], new_rank, data)
    lowrank_final_loss = compute_new_rank_loss(final_model, final_model.layers[layer_ind], new_rank, data)
    print("Low Rank Initial Loss", lowrank_initial_loss)
    print("Low Rank Final Loss", lowrank_final_loss)

    initial_loss_drop_pct = (lowrank_initial_loss - baseline_initial_loss) / baseline_initial_loss
    final_loss_drop_pct = (lowrank_final_loss - baseline_final_loss) / baseline_final_loss
    print("Initial Loss Drop", initial_loss_drop_pct * 100, "%")
    print("Final Loss Drop", final_loss_drop_pct * 100, "%")

    return (final_loss_drop_pct < 0) or (initial_loss_drop_pct > final_loss_drop_pct)

def compute_new_rank_loss(
    model: models.Model,
    layer: low_rank_layer.LowRankLayer,
    new_rank: int,
    data: Tuple[np.ndarray, np.ndarray],
):
    curr_rank = layer.rank
    curr_weights = layer.get_weights()
    layer.set_rank(new_rank)
    loss = evaluate_model_loss(model, data)
    layer.set_rank(curr_rank)
    layer.set_weights(curr_weights)
    model._reset_compile_cache()
    return loss

def main():
    (x, y), val_data = data.load_data(
            "cifar10"
        )
    X_SHAPE = x.shape[1:]
    Y_SHAPE = y.shape[-1]

    # Subset of Data
    x = x[:, :, :, :]
    y = y[:, :]

    optimize_ranks(
        model=get_unoptimized_lr__model(x.shape[1:], y.shape[-1]),
        model_copy=get_unoptimized_lr__model(x.shape[1:], y.shape[-1]),
        data=(x, y)
    )
    

if __name__ == "__main__":
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        parser = argparse.ArgumentParser(
            description="Run multiple low rank training runs"
        )
        parser.add_argument(
            "--gpu",
            default=0,
            type=int,
            choices=range(len(gpus)),
            required=True,
            help="The GPU to use for this experiment. This should be an integer ranging from 0 to "
            "num_gpu - 1",
        )
        args = parser.parse_args()
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            tf.config.set_visible_devices(gpus[args.gpu], "GPU")
        except RuntimeError as e:
            print(e)
    main()
