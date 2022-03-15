import argparse
import datetime
import itertools
import json
import os
import random
from typing import List

import lowrank
import lowrank.low_rank_layer
import numpy as np
import tensorflow as tf
from tensorflow.keras import losses, metrics, optimizers

from . import data
from . import model


class UpdateConvRanksExperiment:
    def __init__(
        self,
        initial_ranks: List[int],
        new_ranks: List[int],
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
        self.model = model.get_vary_conv_rank_model(
            self.x_train.shape[1:], self.y_train.shape[-1], initial_ranks=initial_ranks
        )
        self.model.compile(
            optimizer=optimizers.Adam(0.0001),
            loss=losses.CategoricalCrossentropy(),
            metrics=[metrics.CategoricalAccuracy()],
            run_eagerly=True,
        )
        self.results = {
            "initial_ranks": initial_ranks,
            "new_ranks": new_ranks,
            "rank_update_epoch": rank_update_epoch,
            "total_epochs": total_epochs,
        }

        print("Starting training")
        self._record_fit(rank_update_epoch)

        print("Evaluating pre update")
        self._eval_model(self.x_train, self.y_train, "pre_update")
        self._eval_model(self.x_test, self.y_test, "pre_update_val")

        print("Updating rank")
        self._set_rank(new_ranks)

        print("Evaluating post update")
        self._eval_model(self.x_train, self.y_train, "post_update")
        self._eval_model(self.x_test, self.y_test, "post_update_val")

        print("Continuing training")
        self._record_fit(total_epochs - rank_update_epoch)

    def _eval_model(self, x: np.ndarray, y: np.ndarray, eval_pref: str):
        eval_res = self.model.evaluate(x, y, batch_size=512)
        for metric, value in zip(self.model.metrics_names, eval_res):
            self.results[f"{eval_pref}_{metric}"] = value

    def _set_rank(self, new_ranks: List[int]):
        i = 0
        for layer in self.model.layers:
            if isinstance(layer, lowrank.low_rank_layer.LowRankLayer):
                layer.set_rank(new_ranks[i])
                i += 1

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
    for _ in range(1):
        initial_ranks = [-1, -1, -1, -1, -1]
        new_ranks = random.choice(
            list(
                itertools.product(
                    [-1],
                    [88],
                    [251],
                    [504],
                    [9],
                )
            )
        )
        for update_epoch in [0, 1, 2, 5]:
            print(f"Setting to rank {new_ranks} on epoch {update_epoch}")
            time_str = datetime.datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
            name = f"{str(new_ranks)}_{update_epoch}_{time_str}.json"
            save_loc = os.path.join("vary_conv_ranks_results/", name)
            experiment = UpdateConvRanksExperiment(
                initial_ranks=initial_ranks,
                new_ranks=list(new_ranks),
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
