"""
Experiment to compare various pruners.
"""

import argparse
import pathlib
import random

import tensorflow as tf
from tensorflow.keras import callbacks, losses, metrics, models, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers

import lowrank_experiments.data
import lowrank_experiments.model
from lowrank.low_rank_layer import LowRankLayer
from lowrank.pruners import (
    PruningScope,
    alignment_pruner_gradient_based,
    alignment_pruner_loss_based,
    mag_pruner,
    snip_pruner,
    weight_mag_pruner,
)


def calc_num_weights(model: models.Model) -> int:
    """
    Calculates the number of trainable weights in a model
    """
    num_weights = 0
    for layer in model.layers:
        if isinstance(layer, LowRankLayer):
            if layer.full_rank_mode:
                num_weights += tf.size(layer.kernel_w)
            else:
                sparsity = float(tf.reduce_sum(layer.mask)) / float(tf.size(layer.mask))
                if layer.svd_masking_mode:
                    u, v = layer.kernel_uv
                    num_weights += (float(tf.size(u)) + float(tf.size(v))) * sparsity
                else:
                    assert (
                        layer.weight_masking_mode
                    ), "Layer must be in weight masking mode"
                    num_weights += float(tf.size(layer.kernel_w)) * sparsity
    return int(num_weights)


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
    # taken from https://github.com/geifmany/cifar-vgg/blob/master/cifar10vgg.py
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        # zca_whitening=False,  # apply ZCA whitening
        # rotation_range=15,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=4,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=4,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False,
    )  # randomly flip images
    datagen.fit(x_train)

    model = lowrank_experiments.model.get_model(
        args.model,
        x_train.shape[1:],
        y_train.shape[1],
        weight_decay=args.l2,
    )
    model.compile(
        optimizer=optimizers.SGD(args.lr, decay=5.0e-4, momentum=0.9, nesterov=False),
        # optimizer=optimizers.Adam(args.lr),
        loss=losses.CategoricalCrossentropy(),
        metrics=[
            metrics.CategoricalAccuracy(),
            metrics.CategoricalCrossentropy(),
        ],
    )
    model.fit(
        datagen.flow(x_train, y_train, batch_size=args.batch_size),
        epochs=args.prune_epoch,
        validation_data=(x_test, y_test),
        callbacks=[
            callbacks.TensorBoard(log_dir=tensorboard_log_dir),
            callbacks.LearningRateScheduler(
                lambda epoch: args.lr * (0.1 ** (epoch // 30))
            ),
        ],
    )

    print("Before pruning:")
    loss, acc, cross_entropy = model.evaluate(x_test, y_test)

    with tensorboard_metrics_writer.as_default(step=args.prune_epoch - 1):
        pre_prune_size = calc_num_weights(model)
        print(f"{pre_prune_size =}")
        tf.summary.scalar(name="model_size", data=pre_prune_size)
        tf.summary.scalar(name="preprune_loss", data=loss)
        tf.summary.scalar(name="preprune_acc", data=acc)
        tf.summary.scalar(name="preprune_cross_entropy", data=cross_entropy)

    # prune
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
            data=(x_train[:256], y_train[:256]),
            batch_size=args.batch_size,
            loss=losses.CategoricalCrossentropy(),
        )
    elif args.pruner == "Alignment_Loss":
        pruner = alignment_pruner_loss_based.AlignmentPrunerLossBased(
            model=model,
            scope=args.pruning_scope,
            sparsity=args.sparsity,
            data=(x_train, y_train),
            batch_size=args.batch_size,
            prune_iterations=args.prune_iterations
        )
    elif args.pruner == "Alignment_Gradient":
        pruner = alignment_pruner_gradient_based.AlignmentPrunerGradientBased(
            model=model,
            scope=args.pruning_scope,
            sparsity=args.sparsity,
            data=(x_train, y_train),
            loss=losses.CategoricalCrossentropy(reduction='sum'),
            batch_size=args.batch_size,
            prune_iterations=args.prune_iterations
        )
    elif args.pruner == "WeightMagnitude":
        pruner = weight_mag_pruner.WeightMagPruner(
            model=model,
            scope=args.pruning_scope,
            sparsity=args.sparsity,
            data=(x_train, y_train),
            batch_size=args.batch_size,
        )
    else:
        raise ValueError(f"Invalid pruner: {args.pruner}")

    pruner.prune()

    print("After pruning")
    loss, acc, cross_entropy = model.evaluate(x_test, y_test)

    with tensorboard_metrics_writer.as_default(step=args.prune_epoch):
        post_prune_size = calc_num_weights(model)
        print(f"{post_prune_size =}")
        tf.summary.scalar(name="model_size", data=post_prune_size)
        tf.summary.scalar(name="postprune_loss", data=loss)
        tf.summary.scalar(name="postprune_acc", data=acc)
        tf.summary.scalar(name="postprune_cross_entropy", data=cross_entropy)

    # Train just the batch norm layers for a bit
    for layer in model.layers:
        layer.trainable = isinstance(layer, layers.BatchNormalization)
    model.fit(
        datagen.flow(x_train, y_train, batch_size=args.batch_size),
        epochs=args.prune_epoch + 2,
        validation_data=(x_test, y_test),
        initial_epoch=args.prune_epoch,
        callbacks=[
            callbacks.TensorBoard(log_dir=tensorboard_log_dir),
            callbacks.LearningRateScheduler(
                lambda epoch: args.lr / 4 * (0.1 ** (epoch // 80))
            ),
        ],
    )
    for layer in model.layers:
        layer.trainable = True

    model.fit(
        datagen.flow(x_train, y_train, batch_size=args.batch_size),
        epochs=args.total_epochs,
        validation_data=(x_test, y_test),
        initial_epoch=args.prune_epoch,
        callbacks=[
            callbacks.TensorBoard(log_dir=tensorboard_log_dir),
            callbacks.LearningRateScheduler(
                lambda epoch: args.lr / 4 * (0.1 ** (epoch // 60))
            ),
        ],
    )

    print("End of training")
    model.evaluate(x_test, y_test)


PRUNERS = ["Magnitude", "SNIP", "Alignment_Loss", "Alignment_Gradient", "WeightMagnitude"]
DATASETS = ["cifar10", "cifar100"]
PRUNING_SCOPES = ["global", "local"]
MODELS = ["default", "vgg11", "vgg16", "vgg16_normal", "vgg19"]

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
    parser.add_argument(
        "--gpu", type=int, help="GPU to run on")
    parser.add_argument("--lr", type=float, help="Learning rate")
    parser.add_argument("--l2", type=float, default=0, help="L2 regularization term")
    parser.add_argument("--model", choices=MODELS, help="Model to run experiments with")
    parser.add_argument("--prune_iterations", type=int, help="Number of iterations to prune over")
    args = parser.parse_args()

    if not args.no_gpu:
        gpus = tf.config.list_physical_devices("GPU")[:4]
        if len(gpus) > 0:
            gpu = random.choice(gpus)
            if args.gpu is not None: # if gpu specified, override random choice
                gpu = gpus[args.gpu]
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
