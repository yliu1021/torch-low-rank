from typing import List

import lowrank.low_rank_layer
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras import losses, metrics, models, optimizers

from .data import load_data
from .models import get_vary_conv_rank_model


def main():
    (x_train, y_train), (x_test, y_test) = load_data("cifar10")
    model = get_vary_conv_rank_model(
        x_train.shape[1:], y_train.shape[-1], initial_ranks=[-1, -1, -1, -1, -1]
    )
    model.compile(
        optimizer=optimizers.Adam(0.0001),
        loss=losses.CategoricalCrossentropy(),
        metrics=[metrics.CategoricalAccuracy()],
    )

    svds = [[s] for s in get_svd_dist(model)]
    steps = [0]
    for _ in range(20):
        num_steps = 200
        model.fit(
            x_train,
            y_train,
            batch_size=64,
            steps_per_epoch=num_steps,
            validation_data=(x_test, y_test),
        )
        steps.append(steps[-1] + num_steps)
        for i, svd in enumerate(get_svd_dist(model)):
            svds[i].append(svd)
    svds = [np.array(s) for s in svds]
    print([s.shape for s in svds])

    fig, axes = plt.subplots(len(svds), sharex="all", figsize=(8, 12))
    for ax, svd in zip(axes, svds):
        ax.violinplot(svd.T, steps, widths=50, showmedians=True, showextrema=True)
    plt.show()


def get_svd_dist(model: models.Sequential) -> List[np.ndarray]:
    svds = []
    for layer in model.layers:
        if not isinstance(layer, lowrank.low_rank_layer.LowRankLayer):
            continue
        _, s, _ = np.linalg.svd(layer.eff_weight(), full_matrices=False)
        svds.append(s)
    return svds


if __name__ == "__main__":
    main()
