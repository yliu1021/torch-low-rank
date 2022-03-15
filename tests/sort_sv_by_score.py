from copy import deepcopy
from enum import Enum
import numpy as np
from tensorflow.keras import losses, metrics, models, optimizers
import tensorflow as tf 
from . import data
from .model import get_unoptimized_lr__model

class SCORING_METHOD(Enum):
    SNIP = 1


def sort_sv_by_score(model: models.Model, layer_ind: int, train_data, scoring_method: SCORING_METHOD):
    if scoring_method != SCORING_METHOD.SNIP:
        raise NotImplementedError()

    train_x, train_y = train_data
    layer = model.layers[layer_ind]

    # Ensure weights are saved in SVD form
    if layer.rank == -1:
        layer.set_rank(layer.max_rank)

    # Compute score for each singular vector
    u, v = layer.kernels[layer.rank]
    scores = []
    original_v = deepcopy(v)
    for i in range(v.shape[0]):
        new_v = None
        if i != 0:
            new_v = tf.concat(values=[v[:i,:], tf.zeros_like(v[i:i+1,:]), v[i+1:,:]], axis=0)
        else:
            new_v = tf.concat(values=[tf.zeros_like(v[i:i+1,:]), v[i+1:,:]], axis=0)
        v.assign(new_v)
        new_loss = model.evaluate(train_x, train_y)[0]
        scores.append(-1 * new_loss)
        v.assign(original_v)

    # Sort by score
    ranking = np.argsort(scores)
    v.assign(tf.concat(values=[v[i:i+1,:] for i in ranking], axis=0))
    u.assign(tf.transpose(tf.concat(values=[tf.transpose(u)[i:i+1,:] for i in ranking], axis=0)))

    # Sanity Check! Checks if top ranked singular vector is placed at top now
    assert(original_v[ranking[0]:ranking[0]+1, :] == v[0:1, :])

if __name__ == "__main__":

    gpus = tf.config.list_physical_devices("GPU")
    tf.config.set_visible_devices(gpus[0], "GPU")

    (x, y), val_data = data.load_data(
            "cifar10"
        )
    x = x[:128, :, :, :]
    y = y[:128, :]

    model = get_unoptimized_lr__model(x.shape[1:], y.shape[-1])
    model.compile(
        optimizer=optimizers.Adam(0.0001),
        loss=losses.CategoricalCrossentropy(),
        metrics=[metrics.CategoricalAccuracy()],
        run_eagerly=True,
    )
    sort_sv_by_score(model=model, layer_ind=9, train_data=(x,y), scoring_method=SCORING_METHOD.SNIP)


