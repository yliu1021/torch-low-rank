import numpy as np
from tensorflow.keras import optimizers, losses, metrics

import data
import lowrank
import model


class UpdateRankExperiment:
    def __init__(
        self,
        initial_rank: int,
        new_rank: int,
        rank_update_epoch: int,
        total_epochs: int,
        noise: float = 0.0
    ):
        if total_epochs < rank_update_epoch:
            raise ValueError(
                "Total epochs must be greater than or equal to rank update epoch"
            )
        (self.x_train, self.y_train), (self.x_test, self.y_test) = data.load_data(
            "cifar10", noise=noise
        )
        self.model = model.get_model(
            self.x_train.shape[1:], self.y_train.shape[-1], rank=initial_rank
        )
        self.model.compile(
            optimizer=optimizers.Adam(0.0001),
            loss=losses.CategoricalCrossentropy(),
            metrics=[metrics.CategoricalAccuracy()],
        )
        self.results = {
            "initial_rank": initial_rank,
            "new_rank": new_rank,
            "rank_update_epoch": rank_update_epoch,
            "total_epochs": total_epochs,
            "noise": noise
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
        self.model._reset_compile_cache()  # needed to reset computation graph

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
