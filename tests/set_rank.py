import json
import os

import data
import model
import numpy as np
from tensorflow.keras import losses, metrics, optimizers

import lowrank


class UpdateRankExperiment:
    def __init__(
        self,
        initial_rank: int,
        new_rank: int,
        rank_update_epoch: int,
        total_epochs: int,
    ):
        if total_epochs < rank_update_epoch:
            raise ValueError("Total epochs must be greater than or equal to rank update epoch")
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
        self.model.build(input_shape=self.x_train.shape[1:])

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
    ranks = [-1, 160, 60, 10, 5, 1]
    for initial_ranks in ranks:
        for new_rank in ranks:
            if new_rank == initial_ranks:
                epochs = [0]
            else:
                epochs = [0, 1, 2, 3]
            for update_epoch in epochs:
                print(f"Starting experiment: {initial_ranks} {new_rank} {update_epoch}")
                name = f"{initial_ranks}_{new_rank}_{update_epoch}.json"
                save_loc = os.path.join("set_rank_results", name)
                if os.path.exists(save_loc):
                    print("Experiment already done, skipping")
                    continue
                experiment = UpdateRankExperiment(
                    initial_rank=-1 if initial_ranks == 256 else initial_ranks,
                    new_rank=new_rank,
                    rank_update_epoch=update_epoch,
                    total_epochs=50,
                )
                with open(save_loc, "w") as result_file:
                    json.dump(experiment.results, result_file)


if __name__ == "__main__":
    main()
