'''
Evalute Singular Vector Ranking Methods = Looking to determine which method of ranking singular vectors allows us to prune while losing minimal accuracy, highest pruning ration and can be evaluated early in training
'''

import os, sys

BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_PATH)

import argparse
import tensorflow as tf 
from tensorflow.keras import losses, metrics, models, optimizers
from lowrank.low_rank_layer import LowRankLayer

from tests.sort_sv_by_score import SCORING_METHOD, sort_sv_by_score
from tests import data
from tests import models

def main(args):
    '''
    Current Logic (Model - Low Rank Layers with Initial Rank = Max Rank)
    - Traing pre-pruning for given number of epochs
    - For each low rank layer, sort singular vectors by method and retain pruning_ratio of the singular vectors
    - Train low rank model for remaining epochs
    '''
    gpus = tf.config.list_physical_devices("GPU")
    tf.config.set_visible_devices(gpus[0], "GPU")

    (train_x, train_y), (val_x, val_y) = data.load_data(
            "cifar10"
        )
    
    if args.fast:
        train_x = train_x[:128, :, :, :]
        train_y = train_y[:128, :]

    model = models.get_unoptimized_lr__model(train_x.shape[1:], train_y.shape[-1])
    model.compile(
        optimizer=optimizers.Adam(0.0001),
        loss=losses.CategoricalCrossentropy(),
        metrics=[metrics.CategoricalAccuracy()],
        run_eagerly=True,
    )

    train_metrics = model.fit(train_x, train_y, batch_size=args.batch_size, epochs=args.pre_prune_epochs)
    for metric, values in train_metrics.history.items():
        print(metric, values)

    for i, layer in enumerate(model.layers):
        if issubclass(type(layer), LowRankLayer):
            sort_sv_by_score(
                model=model,
                layer_ind=i,
                train_data=(train_x, train_y),
                scoring_method=args.method
            )
            layer.set_rank(new_rank=int(layer.max_rank * args.pruning_ratio), reuse_existing_kernels=True)
            print("Layer " + str(i) + " New Rank: " + str(int(layer.max_rank * args.pruning_ratio)))
    
    train_metrics = model.fit(train_x, train_y, batch_size=args.batch_size, epochs=(args.total_epochs - args.pre_prune_epochs))
    for metric, values in train_metrics.history.items():
        print(metric,":", values)

    val_metrics = model.evaluate(val_x, val_y, batch_size=args.batch_size)
    for metric, value in zip(model.metrics_names, val_metrics):
        print(f"val_{metric}:", value)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate singular vector rankings")
    parser.add_argument('--method', type=int, help="Method to score ranks by: " + str(list(SCORING_METHOD)))
    parser.add_argument('--pre_prune_epochs', type=int, help="Number of epochs to train for before pruning")
    parser.add_argument('--total_epochs', type=int, help="Total number of epochs to train for")
    parser.add_argument('--fast', action='store_true', default=False)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--pruning_ratio', type=float, help="Ratio of singular vectors pruned from each layer")

    args = parser.parse_args()
    args.method = SCORING_METHOD(args.method)

    main(args)