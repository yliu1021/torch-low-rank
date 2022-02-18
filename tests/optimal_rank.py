import argparse
import enum
from pprint import pprint
from typing import Tuple

import numpy as np
import tensorflow as tf
from lowrank import low_rank_layer
from tensorflow.keras import losses, metrics, models, optimizers
from tensorflow.python.framework.ops import Tensor
from tensorflow.python.training.tracking import base

import data
from model import get_unoptimized_lr__model


class OptimizationMethod(enum.Enum):
    BACK_TO_FRONT = enum.auto()
    FRONT_TO_BACK = enum.auto()

class RankOptimizer():
    def __init__(self, 
        model: models.Model,
        data: Tuple[np.ndarray, np.ndarray],
        layer_order: OptimizationMethod = OptimizationMethod.BACK_TO_FRONT,
        epochs: int = 5,
    ):

        self.model = model
        self.data = data 
        self.layer_order = layer_order 
        self.epochs = epochs 

        self.optimal_ranks = {}
        self.saved_weights = []

    def run(self):
        """
        Optimizes the ranks of a low rank model by monitoring training loss before and after
        compression at various epochs
        """
        num_layers = len(self.model.layers)
        if self.layer_order is OptimizationMethod.BACK_TO_FRONT:
            layer_inds = reversed(range(num_layers))
        elif self.layer_order is OptimizationMethod.FRONT_TO_BACK:
            layer_inds = range(num_layers)
        else:
            raise NotImplementedError(f"Layer order: {self.layer_order} not supported")
        
        '''
        Train Model
        ''' 
        print("Training Model")
        x, y = self.data
        self.model.compile(
                optimizer=optimizers.Adam(0.0001),
                loss=losses.CategoricalCrossentropy(),
                metrics=[metrics.CategoricalAccuracy()],
                run_eagerly=True,
            )

        # Train
        for _ in range(self.epochs):
            self.model.fit(x, y, batch_size=64)
            self.saved_weights.append(self.model.get_weights())

        print("Training Complete.")
        
        '''
        Iterate through layers to determine optimal ranks
        '''
        for i in layer_inds:
            if not isinstance(self.model.layers[i], low_rank_layer.LowRankLayer):
                continue

            baseline_losses = []

            print("Evaluating Baseline Losses")
            for j in range(self.epochs):
                # Reset ranks to full to set weights correctly

                baseline_losses.append(self.evaluate_model_loss())
            
            print("Determining Optimal Rank for Layer " + str(i))

            optimal_rank = self.optimize_rank_of_layer(layer_ind=i, baseline_losses=baseline_losses)
            self.optimal_ranks[i] = optimal_rank
            
            print("Optimal Rank for Layer " + str(i) + " = " + str(optimal_rank))

        print("Optimal Ranks: ")
        pprint(self.optimal_ranks)

        return self.model

    def evaluate_model_loss(self):
        for layer_ind in self.optimal_ranks.keys():
            self.model.layers[layer_ind].set_rank(self.optimal_ranks[layer_ind])
        self.model._reset_compile_cache()
        x, y = self.data
        losses = self.model.evaluate(x, y, batch_size=64)
        return losses[0]

    def optimize_rank_of_layer(
        self,
        layer_ind: int,
        baseline_losses
    ):
        """
        Optimizes the rank of just 1 layer specified by `layer_ind`
        """
        if not isinstance(self.model.layers[layer_ind], low_rank_layer.LowRankLayer):
            raise ValueError(f"Layer must be a low rank layer. Got a {type(self.model.layers[layer_ind])} instead")
        
        print("Baseline Losses")
        print(baseline_losses)

        max_insufficient_rank = 1
        min_sufficient_rank = self.max_rank(self.model.layers[layer_ind])
        while(min_sufficient_rank > max_insufficient_rank + 1):
            print("Min Sufficient Rank", min_sufficient_rank)
            print("Max Insufficient Rank", max_insufficient_rank)
            new_rank = int((min_sufficient_rank + max_insufficient_rank) / 2)
            if self.is_sufficient_rank(new_rank, layer_ind, baseline_losses):
                min_sufficient_rank = new_rank
            else:
                max_insufficient_rank = new_rank

        return min_sufficient_rank

    def max_rank(self, lr_layer: low_rank_layer):
        shape = lr_layer.eff_weight().shape
        if not len(shape) == 2:
            raise ValueError(f"Layer's weights must be formatted as a 2D Tensor")
        return min(shape)

    def is_sufficient_rank(self,
        new_rank: int,
        layer_ind: int, 
        baseline_losses
    ):
        lowrank_losses = []
        loss_drop_pcts = []

        for i in range(len(baseline_losses)):
            self.set_model_weights(self.saved_weights[i])
            lowrank_losses.append(self.compute_new_rank_loss(self.model.layers[layer_ind], new_rank))
            print("Low Rank Loss at Epoch", i+1, lowrank_losses[-1])
            loss_drop_pcts.append((lowrank_losses[-1] - baseline_losses[-1]) / baseline_losses[-1])
        
        print("Loss Drop Percentages: ")
        for loss_drop_pct in loss_drop_pcts:
            print(loss_drop_pct)

        print("Initial Loss Drop", loss_drop_pcts[0] * 100, "%")
        print("Final Loss Drop", loss_drop_pcts[-1] * 100, "%")

        return (loss_drop_pcts[-1] < 0) or (loss_drop_pcts[0] > loss_drop_pcts[-1])

    def compute_new_rank_loss(self, 
        layer: low_rank_layer.LowRankLayer,
        new_rank: int
    ):
        curr_rank = layer.rank
        curr_weights = layer.get_weights()
        layer.set_rank(new_rank)
        loss = self.evaluate_model_loss()
        layer.set_rank(curr_rank)
        layer.set_weights(curr_weights)
        self.model._reset_compile_cache()
        return loss

    def set_model_weights(self, weights):
        for layer in self.model.layers:
            if isinstance(layer, low_rank_layer.LowRankLayer):
                layer.set_rank(-1)
        self.model.set_weights(weights)
    
    def train_optimal_rank_model(self, start_epoch, end_epoch, val_data):
        # Validation
        for i in range(len(self.model.layers)):
            if isinstance(self.model.layers[i], low_rank_layer.LowRankLayer):
                if i not in self.optimal_ranks:
                    raise Exception("Optimal ranks not found for all low rank layers yet. Cannot train optimized model.")
        if start_epoch > self.epochs:
            raise Exception("Cannot start training at this epoch. Rank Optimizer did not train so far.")
        
        self.set_model_weights(self.saved_weights[start_epoch-1])
        for layer_ind in self.optimal_ranks.keys():
            self.model.layers[layer_ind].set_rank(self.optimal_ranks[layer_ind])

        x, y = self.data
        val_x, val_y = val_data
        train_losses = []
        train_acc = []
        test_losses = []
        test_acc = []
        for _ in range(end_epoch-start_epoch):
            # Training
            train_evl = self.model.fit(x, y, batch_size=64)
            train_losses.append(train_evl.history["loss"])
            train_acc.append(train_evl.history["categorical_accuracy"])
            # Validation Metrics
            test_evl = self.model.evaluate(val_x, val_y, batch_size=64)
            test_losses.append(test_evl[0])
            test_acc.append(test_evl[1])
        return train_losses, train_acc, test_losses, test_acc
            
def main():
    (x, y), val_data = data.load_data(
            "cifar10"
        )
    X_SHAPE = x.shape[1:]
    Y_SHAPE = y.shape[-1]

    # Subset of Data
    x = x[:, :, :, :]
    y = y[:, :]

    rank_optimizer = RankOptimizer(
        model=get_unoptimized_lr__model(x.shape[1:], y.shape[-1]),
        data=(x, y)
    )
    rank_optimizer.run()
    pprint(rank_optimizer.train_optimal_rank_model(start_epoch=5, end_epoch=50, val_data=val_data))
    

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
