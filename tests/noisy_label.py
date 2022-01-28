import argparse
import datetime
import itertools
import json
import os
import random

import tensorflow as tf

from set_rank_experiment import UpdateRankExperiment


def main():
    for _ in range(500):
        update_epoch, noise = random.choice(
            list(itertools.product([1, 2, 5, 10, 25], [0, 0.2, 0.5, 0.7]))
        )
        print(
            f"Update epoch: {update_epoch}, Noise: {noise}"
        )
        time_str = datetime.datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
        name = f"{update_epoch}_{noise}_{time_str}.json"
        save_loc = os.path.join("noisy_label_results", name)
        experiment = UpdateRankExperiment(
            initial_rank=200,
            new_rank=14,
            rank_update_epoch=update_epoch,
            total_epochs=50,
            noise=noise
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
