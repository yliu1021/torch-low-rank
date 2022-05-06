"""
Tests linear mode connectivity in low rank training
"""

import argparse
import pathlib
import random

import tensorflow as tf
from tensorflow.keras import callbacks, losses, metrics, models, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import lowrank_experiments.data_loader
import lowrank_experiments.model
from lowrank.pruners import (
    PruningScope,
    alignment_pruner_loss_based,
    mag_pruner,
    snip_pruner,
    weight_mag_pruner,
)


def main(
    dataset: str, fast: bool, model_name: str, l2: float, lr: float, batch_size: int
):
    (x_train, y_train), (x_test, y_test) = lowrank_experiments.data.load_data(
        dataset, fast
    )
    # taken from https://github.com/geifmany/cifar-vgg/blob/master/cifar10vgg.py
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=15,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False,
    )  # randomly flip images
    datagen.fit(x_train)

    model = lowrank_experiments.model.get_model(
        model_name,
        x_train.shape[1:],
        y_train.shape[1],
        initial_ranks=None,
        weight_decay=l2,
    )
    model.compile(
        optimizer=optimizers.SGD(lr, decay=1e-6, momentum=0.9, nesterov=True),
        loss=losses.CategoricalCrossentropy(),
        metrics=[
            metrics.CategoricalAccuracy(),
            metrics.CategoricalCrossentropy(),
        ],
    )

    eff_weights = []

    for i in range(10):
        print(f"Epoch {i}")
        model.fit(
            datagen.flow(x_train, y_train, batch_size=batch_size),
            validation_data=(x_test, y_test),
            callbacks=[
                callbacks.LearningRateScheduler(
                    lambda epoch: lr * (0.5 ** (epoch // 20))
                ),
            ],
            initial_epoch=i,
            epochs=i + 1,
        )
        model.save(f"model_save_{i}")


DATASETS = ["cifar10", "cifar100"]
MODELS = ["default", "vgg11", "vgg16", "vgg16_normal"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Linear mode connectivity of low rank training"
    )
    parser.add_argument("--dataset", choices=DATASETS, help="Choice of dataset")
    parser.add_argument(
        "--fast",
        action="store_true",
        default=False,
        help="Enable to run fast mode. \
            Fast mode subsets the dataset. To be used for verifying code",
    )
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--lr", type=float, help="Learning rate")
    parser.add_argument("--l2", type=float, default=0, help="L2 regularization term")
    parser.add_argument("--model", choices=MODELS, help="Model to run experiments with")
    args = parser.parse_args()
    main(
        dataset=args.dataset,
        fast=args.fast,
        model_name=args.model,
        l2=args.l2,
        lr=args.lr,
        batch_size=args.batch_size,
    )
