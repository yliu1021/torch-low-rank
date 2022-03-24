"""
Experiment to compare various pruners.
"""

import argparse
import pathlib
from random import randint

import tensorflow as tf
from tensorflow.keras import callbacks, losses, metrics, optimizers

import lowrank_experiments.data
import lowrank_experiments.model
from lowrank.pruners import PruningScope, alignment_pruner, mag_pruner, snip_pruner


def main(args):
    """
    Main method that runs the experiment
    """
    tensorboard_log_dir = pathlib.Path("./logs_set_rank")
    tensorboard_log_dir.mkdir(exist_ok=True)  # make root logging directory

    (x_train, y_train), (x_test, y_test) = lowrank_experiments.data.load_data(
        args.dataset, args.fast
    )
    model = lowrank_experiments.model.get_lr_model(
        x_train.shape[1:],
        num_classes=y_train.shape[1],
        initial_ranks=[-1, -1, -1, -1, -1],  # start full rank
    )

    model.compile(
        optimizer=optimizers.RMSprop(0.001),
        loss=losses.CategoricalCrossentropy(),
        metrics=[metrics.CategoricalAccuracy()],
    )

    model.fit(
        x_train,
        y_train,
        batch_size=args.batch_size,
        epochs=args.prune_epoch,
        validation_data=(x_test, y_test),
        callbacks=[callbacks.TensorBoard(log_dir=tensorboard_log_dir)],
    )

    print("Before pruning:")
    model.evaluate(x_test, y_test)

    pruner = None
    if args.pruner == "Magnitude":
        pruner = mag_pruner.MagPruner(
            model=model,
            scope=args.pruning_scope,
            sparsity=args.sparsity,
            batch_size=args.batch_size,
        )
    elif args.pruner == "SNIP":
        pruner = snip_pruner.SnipPruner(
            model=model,
            scope=args.pruning_scope,
            sparsity=args.sparsity,
            data=(x_train[:64], y_train[:64]),
            batch_size=args.batch_size,
            loss=losses.CategoricalCrossentropy(),
        )
    elif args.pruner == "Alignment":
        pruner = alignment_pruner.AlignmentPruner(
            model=model,
            scope=args.pruning_scope,
            sparsity=args.sparsity,
            data=(x_train, y_train),
            batch_size=args.batch_size,
        )

    pruner.prune()

    print("After pruning")
    model.evaluate(x_test, y_test)
    model.fit(
        x_train,
        y_train,
        batch_size=args.batch_size,
        epochs=args.total_epochs,
        validation_data=(x_test, y_test),
        initial_epoch=args.prune_epoch,
        callbacks=[callbacks.TensorBoard(log_dir=tensorboard_log_dir)],
    )

    print("End of training")
    model.evaluate(x_test, y_test)


PRUNERS = ["Magnitude", "SNIP", "Alignment"]
DATASETS = ["cifar10", "cifar100"]
PRUNING_SCOPES = ["global", "local"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate singular vector rankings")
    parser.add_argument("--dataset", choices=DATASETS, help="Choice of dataset")
    parser.add_argument("--pruner", choices=PRUNERS, help="Choice of pruning algorithm")
    parser.add_argument("--prune_epoch", type=int, help="Epoch to prune at")
    parser.add_argument(
        "--total_epochs", type=int, help="Total number of epochs to train for"
    )
    parser.add_argument("--batch_size", type=int)
    parser.add_argument(
        "--sparsity", type=float, help="Percentage of singular vectors to be pruned"
    )
    parser.add_argument(
        "--pruning_scope",
        choices=PRUNING_SCOPES,
        help="Scope to rank singular vectors (global or layer wise)",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        default=False,
        help="Enable to run fast mode. \
        Fast mode subsets the dataset. To be used for verifying code",
    )
    parser.add_argument(
        "--no_gpu", action="store_true", default=False, help="Disable GPU"
    )
    args = parser.parse_args()

    if not args.no_gpu:
        gpus = tf.config.list_physical_devices("GPU")
        tf.config.set_visible_devices(gpus[randint(0, len(gpus) - 1)], "GPU")

    # Preprocess arguments
    if args.pruning_scope == "global":
        args.pruning_scope = PruningScope.GLOBAL
    elif args.pruning_scope == "local":
        args.pruning_scope = PruningScope.LOCAL
    else:
        raise argparse.ArgumentError(argument=None, message="Unsupported pruning scope")

    main(args)
