"""
Experiment to compare various pruners.
"""

import argparse
import pathlib
import random

import tensorflow as tf
from tensorflow.keras import callbacks, losses, metrics, models, optimizers

import lowrank_experiments.data
import lowrank_experiments.model
from lowrank.pruners import (
    PruningScope,
    alignment_pruner,
    mag_pruner,
    snip_pruner,
    weight_mag_pruner,
)


def calc_num_weights(model: models.Model) -> int:
    """
    Calculates the number of trainable weights in a model
    """
    num_weights = 0
    for weight in model.trainable_weights:
        num_weights += tf.size(weight)
    return num_weights


def main(args):
    """
    Main method that runs the experiment
    """
    tensorboard_log_dir = pathlib.Path("./logs_set_rank")
    tensorboard_log_dir.mkdir(exist_ok=True)  # make root logging directory
    tensorboard_metrics_writer = tf.summary.create_file_writer(
        str(tensorboard_log_dir / "metrics")
    )

    (x_train, y_train), (x_test, y_test) = lowrank_experiments.data.load_data(
        args.dataset, args.fast
    )

    if args.model == "default":
        model = lowrank_experiments.model.get_lr_model(
            x_train.shape[1:], num_classes=y_train.shape[1], initial_ranks=None
        )
    elif args.model == "vgg11":
        print(args)
        model = lowrank_experiments.model.get_lr_vgg11(
            x_train.shape[1:],
            num_classes=y_train.shape[1],
            initial_ranks=None,
            weight_decay=args.l2,
        )
    elif args.model == "vgg16":
        model = lowrank_experiments.model.get_lr_vgg16(
            x_train.shape[1:],
            num_classes=y_train.shape[1],
            initial_ranks=None,
            weight_decay=args.l2,
        )
    elif args.model == "vgg16_normal":
        model = lowrank_experiments.model.get_vgg16(
            x_train.shape[1:], num_classes=y_train.shape[1], initial_ranks=None
        )
    else:
        raise NotImplementedError(args.model + " is not supported currently.")

    model.compile(
        optimizer=optimizers.SGD(args.lr, momentum=0.9),
        # optimizer=optimizers.Adam(args.lr),
        loss=losses.CategoricalCrossentropy(),
        metrics=[metrics.CategoricalAccuracy()],
    )

    model.fit(
        x_train,
        y_train,
        batch_size=args.batch_size,
        epochs=args.prune_epoch,
        validation_data=(x_test, y_test),
        callbacks=[
            callbacks.TensorBoard(log_dir=tensorboard_log_dir),
            callbacks.ReduceLROnPlateau(patience=10),
        ],
    )

    print("Before pruning:")
    (loss, acc) = model.evaluate(x_test, y_test)

    with tensorboard_metrics_writer.as_default(step=args.prune_epoch - 1):
        pre_prune_size = calc_num_weights(model)
        tf.summary.scalar(name="model_size", data=pre_prune_size)
        tf.summary.scalar(name="preprune_loss", data=loss)
        tf.summary.scalar(name="preprune_acc", data=acc)

    # prune
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
            data=(x_train[:2048], y_train[:2048]),
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
    elif args.pruner == "WeightMagnitude":
        pruner = weight_mag_pruner.WeightMagPruner(
            model=model,
            scope=args.pruning_scope,
            sparsity=args.sparsity,
            data=(x_train, y_train),
            batch_size=args.batch_size,
        )
    pruner.prune()

    print("After pruning")
    loss, acc = model.evaluate(x_test, y_test)

    with tensorboard_metrics_writer.as_default(step=args.prune_epoch):
        post_prune_size = calc_num_weights(model)
        tf.summary.scalar(name="model_size", data=post_prune_size)
        tf.summary.scalar(name="postprune_loss", data=loss)
        tf.summary.scalar(name="postprune_acc", data=acc)

    model.fit(
        x_train,
        y_train,
        batch_size=args.batch_size,
        epochs=args.total_epochs,
        validation_data=(x_test, y_test),
        initial_epoch=args.prune_epoch,
        callbacks=[
            callbacks.TensorBoard(log_dir=tensorboard_log_dir),
            callbacks.ReduceLROnPlateau(patience=5),
        ],
    )

    print("End of training")
    model.evaluate(x_test, y_test)


PRUNERS = ["Magnitude", "SNIP", "Alignment", "WeightMagnitude"]
DATASETS = ["cifar10", "cifar100"]
PRUNING_SCOPES = ["global", "local"]
MODELS = ["default", "vgg11", "vgg16", "vgg16_normal"]

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
    parser.add_argument("--lr", type=float, help="Learning rate")
    parser.add_argument("--l2", type=float, default=0, help="L2 regularization term")
    parser.add_argument("--model", choices=MODELS, help="Model to run experiments with")
    args = parser.parse_args()

    if not args.no_gpu:
        gpus = tf.config.list_physical_devices("GPU")
        if len(gpus) > 0:
            gpu = random.choice(gpus)
            tf.config.experimental.set_memory_growth(gpu, True)
            tf.config.set_visible_devices(gpu, "GPU")

    # Preprocess arguments
    if args.pruning_scope == "global":
        args.pruning_scope = PruningScope.GLOBAL
    elif args.pruning_scope == "local":
        args.pruning_scope = PruningScope.LOCAL
    else:
        raise argparse.ArgumentError(argument=None, message="Unsupported pruning scope")

    main(args)
