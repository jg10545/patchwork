import numpy as np
import tensorflow as tf

tf.random.set_seed(1)
from patchwork.feature._nnclr import _find_nearest_neighbors, _update_queue, _build_nnclr_training_step


def test_find_nearest_neighbors():
    qs = 100
    N = 10
    d = 5
    Q = tf.nn.l2_normalize(np.random.normal(0, 1, (qs, d)).astype(np.float32), 1)
    z = Q[:N, :]

    nn = _find_nearest_neighbors(z, Q)
    assert nn.shape == z.shape
    assert tf.reduce_sum((nn - z) ** 2).numpy() < 1e-5


def test_update_queue():
    qs = 100
    N = 10
    d = 5
    Q = tf.Variable(tf.nn.l2_normalize(np.random.normal(0, 1, (qs,d)).astype(np.float32), 1))
    z = tf.nn.l2_normalize(np.random.normal(0, 1, (N,d)).astype(np.float32), 1)
    oldshape = Q.shape

    _update_queue(z,Q)
    assert Q.shape == oldshape
    assert (Q[:N,:] == z).numpy().all()

def test_nnclr_training_step():
    N = 3
    queue_size = 17
    d = 5
    D = 7

    Q = tf.Variable(np.random.normal(0, 1, size=(queue_size, d)).astype(np.float32))
    x = np.random.normal(0, 1, size=(N, D)).astype(np.float32)
    y = np.random.normal(0, 1, size=(N, D)).astype(np.float32)

    inpt = tf.keras.layers.Input((D))
    net = tf.keras.layers.Dense(d)(inpt)
    embed_model = tf.keras.Model(inpt,net)

    opt = tf.keras.optimizers.SGD()

    step = _build_nnclr_training_step(embed_model, opt, Q)

    loss_dict = step(x,y)
    assert isinstance(loss_dict, dict)
    for k in ["nt_xent_loss", "l2_loss", "loss", "nce_batch_acc"]:
        assert k in loss_dict


def test_nnclr_training_step_with_weight_decay():
    N = 3
    queue_size = 17
    d = 5
    D = 7

    Q = tf.Variable(np.random.normal(0, 1, size=(queue_size, d)).astype(np.float32))
    x = np.random.normal(0, 1, size=(N, D)).astype(np.float32)
    y = np.random.normal(0, 1, size=(N, D)).astype(np.float32)

    inpt = tf.keras.layers.Input((D))
    net = tf.keras.layers.Dense(d)(inpt)
    embed_model = tf.keras.Model(inpt,net)

    opt = tf.keras.optimizers.legacy.SGD()

    step = _build_nnclr_training_step(embed_model, opt, Q, weight_decay=1e-6)

    loss_dict = step(x,y)
    assert isinstance(loss_dict, dict)
    for k in ["nt_xent_loss", "l2_loss", "loss", "nce_batch_acc"]:
        assert k in loss_dict
