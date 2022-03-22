from tensorflow.keras import losses, metrics, optimizers

import lowrank.pruners
import lowrank_experiments.data
import lowrank_experiments.model
from lowrank.pruners import alignment_pruner, mag_pruner


def main():
    (x_train, y_train), (x_test, y_test) = lowrank_experiments.data.load_data("cifar10")
    model = lowrank_experiments.model.get_lr_model(
        x_train.shape[1:],
        num_classes=y_train.shape[1],
        initial_ranks=[-1, -1, -1, -1, -1],  # start full rank
    )
    model.compile(
        optimizer=optimizers.RMSprop(0.001),
        loss=losses.CategoricalCrossentropy(),
        metrics=[metrics.CategoricalAccuracy()],
    )
    model.fit(
        x_train, y_train, batch_size=128, epochs=5, validation_data=(x_test, y_test)
    )
    print("Before pruning:")
    model.evaluate(x_test, y_test)
    pruner = mag_pruner.MagPruner(
        model, lowrank.pruners.PruningScope.LOCAL, sparsity=0.5
    )
    pruner.prune()
    print("After pruning")
    model.evaluate(x_test, y_test)
    model.fit(
        x_train, y_train, batch_size=128, epochs=10, validation_data=(x_test, y_test)
    )


if __name__ == "__main__":
    main()
