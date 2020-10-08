# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf

from patchwork.feature._rotation import _build_rotation_training_step

def test_build_rotation_training_step():
    inpt = tf.keras.layers.Input((None, None, 3))
    net = tf.keras.layers.Conv2D(2,1)(inpt)
    net = tf.keras.layers.GlobalMaxPool2D()(net)
    net = tf.keras.layers.Dense(4, activation="softmax")(net)
    model = tf.keras.Model(inpt, net)
    opt = tf.keras.optimizers.SGD()
    
    step = _build_rotation_training_step(model, opt)
    
    x = np.zeros((7,16,16,3)).astype(np.float32)
    y = np.zeros(7).astype(np.float64)
    
    lossdict = step(x,y)
    assert isinstance(lossdict, dict)
    loss = lossdict["loss"].numpy()
    assert loss.shape == ()
    assert loss.dtype == np.float32
    