import lowrank
import numpy as np
from tensorflow.keras import losses, metrics, optimizers

from . import data
from . import model


def get_rank(model):
    for layer in model.layers:
        if not isinstance(layer, lowrank.LRDense):
            continue
        eff_weight = layer.eff_weight()
        s = np.linalg.svd(eff_weight, full_matrices=False, compute_uv=False)
        print(f"Eigenvalue min/max/mean: {s.min()}, {s.max()}, {s.mean()}")
        return np.sum(
            s > 1e-7
        )  # since we compute (U @ V) first, there's fp rounding errors


def main():
    (x_train, y_train), (x_test, y_test) = data.load_data("cifar100")
    lr_model = model.get_lr_conv_model(x_train.shape[1:], y_train.shape[-1], rank=-1)
    lr_model.compile(
        optimizer=optimizers.Adam(0.0001),
        loss=losses.CategoricalCrossentropy(),
        metrics=[metrics.CategoricalAccuracy()],
    )

    print(f"Initial rank: {get_rank(lr_model)}")
    lr_model.fit(
        x_train,
        y_train,
        batch_size=256,
        epochs=1,
        validation_data=(x_test, y_test),
    )
    print(f"Rank after training: {get_rank(lr_model)}")
    for layer in lr_model.layers:
        if not isinstance(layer, lowrank.LRDense):
            continue
        layer.set_rank(10)
    print(f"Rank after dropping: {get_rank(lr_model)}")
    lr_model.fit(
        x_train,
        y_train,
        batch_size=256,
        epochs=1,
        validation_data=(x_test, y_test),
    )
    print(f"Final rank: {get_rank(lr_model)}")


if __name__ == "__main__":
    main()
