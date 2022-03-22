import argparse
import datetime
import itertools
import json
import os
import random

import numpy as np
import tensorflow as tf

from set_rank_experiment import UpdateRankExperiment


def main():
    possible_ranks = [int(round(x)) for x in np.logspace(0, np.log10(200), num=5)]
    possible_ranks += [-1]
    for _ in range(500):
        initial_rank, new_rank, update_epoch = random.choice(
            list(itertools.product(possible_ranks, possible_ranks, [1, 2, 5, 10, 25]))
        )
        print(
            f"Initial rank: {initial_rank}, new rank: {new_rank}, update epoch: {update_epoch}"
        )
        time_str = datetime.datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
        name = f"{initial_rank}, {new_rank}_{update_epoch}_{time_str}.json"
        save_loc = os.path.join("set_rank_results_cifar10", name)
        experiment = UpdateRankExperiment(
            initial_rank=initial_rank,
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
