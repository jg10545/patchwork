# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf

from patchwork.feature._simclr_multi import _build_distributed_training_step
from patchwork.feature._simclr import _build_embedding_model


strat = tf.distribute.OneDeviceStrategy("/cpu:0")

with strat.scope():
    # build a tiny FCN for testing
    inpt = tf.keras.layers.Input((None, None, 3))
    net = tf.keras.layers.Conv2D(5,1)(inpt)
    net = tf.keras.layers.MaxPool2D(10,10)(net)
    fcn = tf.keras.Model(inpt, net)


def test_build_simclr_training_step():
    with strat.scope():
        model = _build_embedding_model(fcn, (32,32), 3, 5, 7)
        opt = tf.keras.optimizers.SGD()
    
    step = _build_distributed_training_step(strat, model, opt, 0.1)
    
    x = tf.zeros((4,32,32,3), dtype=tf.float32)
    y = np.array([1,-1,1,-1]).astype(np.int32)
    lossdict = step(x,y)
    
    assert isinstance(lossdict["loss"].numpy(), np.float32)
    # should include loss and average cosine similarity
    assert len(lossdict) == 2
    
    
def test_build_simclr_training_step_with_weight_decay():
    with strat.scope():
        model = _build_embedding_model(fcn, (32,32), 3, 5, 7)
        opt = tf.keras.optimizers.SGD()
    
    step = _build_distributed_training_step(strat, model, opt, 0.1,
                                            weight_decay=1e-6)
    x = tf.zeros((4,32,32,3), dtype=tf.float32)
    y = np.array([1,-1,1,-1]).astype(np.int32)
    lossdict = step(x,y)
    
    # should include total loss, crossent loss, average cosine
    # similarity and L2 norm squared
    assert len(lossdict) == 4