import argparse
import datetime
import itertools
import json
import os
import random

import lowrank
import numpy as np
import tensorflow as tf
from tensorflow.keras import losses, metrics, optimizers

import data
import model


class UpdateRankExperiment:
    def __init__(
        self,
        initial_rank: int,
        new_rank: int,
        rank_update_epoch: int,
        total_epochs: int,
    ):
        if total_epochs < rank_update_epoch:
            raise ValueError(
                "Total epochs must be greater than or equal to rank update epoch"
            )
        (self.x_train, self.y_train), (self.x_test, self.y_test) = data.load_data(
            "cifar10"
        )
        self.model = model.get_model(
            self.x_train.shape[1:], self.y_train.shape[-1], rank=initial_rank
        )
        self.model.compile(
            optimizer=optimizers.Adam(0.0001),
            loss=losses.CategoricalCrossentropy(),
            metrics=[metrics.CategoricalAccuracy()],
            run_eagerly=True,
        )
        self.results = {
            "initial_rank": initial_rank,
            "new_rank": new_rank,
            "rank_update_epoch": rank_update_epoch,
            "total_epochs": total_epochs,
        }

        print("Starting training")
        self._record_fit(rank_update_epoch)

        print("Evaluating pre update")
        self._eval_model(self.x_train, self.y_train, "pre_update")
        self._eval_model(self.x_test, self.y_test, "pre_update_val")

        print("Updating rank")
        self._set_rank(new_rank)

        print("Evaluating post update")
        self._eval_model(self.x_train, self.y_train, "post_update")
        self._eval_model(self.x_test, self.y_test, "post_update_val")

        print("Continuing training")
        self._record_fit(total_epochs - rank_update_epoch)

    def _eval_model(self, x: np.ndarray, y: np.ndarray, eval_pref: str):
        eval_res = self.model.evaluate(x, y, batch_size=512)
        for metric, value in zip(self.model.metrics_names, eval_res):
            self.results[f"{eval_pref}_{metric}"] = value

    def _set_rank(self, new_rank: int):
        for layer in self.model.layers:
            if not isinstance(layer, lowrank.LRDense):
                continue
            layer.set_rank(new_rank)

    def _record_fit(self, epochs: int):
        if epochs == 0:
            return
        train = self.model.fit(
            self.x_train,
            self.y_train,
            batch_size=64,
            epochs=epochs,
            validation_data=(self.x_test, self.y_test),
        )
        for metric, values in train.history.items():
            self.results[metric] = self.results.get(metric, []) + values


def main():
    for _ in range(100):
        new_rank, update_epoch = random.choice(
            list(itertools.product([-1, 160, 50, 10, 1], [1, 2, 3]))
        )
        print(f"Setting to rank {new_rank} on epoch {update_epoch}")
        time_str = datetime.datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
        name = f"{new_rank}_{update_epoch}_{time_str}.json"
        save_loc = os.path.join("set_rank_results_cifar10", name)
        experiment = UpdateRankExperiment(
            initial_rank=-1,
            new_rank=new_rank,
            rank_update_epoch=update_epoch,
            total_epochs=50,
        )
        with open(save_loc, "w") as result_file:
            json.dump(experiment.results, result_file)
            result_file.flush()
        print("Saved")


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
