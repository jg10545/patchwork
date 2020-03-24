# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf

from patchwork.feature._moco import copy_model, exponential_model_update


def test_copy_model():
    orig = tf.keras.applications.VGG16(weights=None, include_top=False)
    clone = copy_model(orig)
    # number of layers should be the same
    assert len(orig.layers) == len(clone.layers)
    # weights should be identical
    assert np.sum((orig.layers[1].get_weights()[0]-clone.layers[1].get_weights()[0])) == 0.


def test_exponenial_model_update():
    test_inpt = np.ones((1,5), dtype=np.float32)
    
    inpt = tf.keras.layers.Input((5))
    net = tf.keras.layers.Dense(4)(inpt)
    mod1 = tf.keras.Model(inpt, net)
    out1 = mod1(test_inpt).numpy()
    
    inpt = tf.keras.layers.Input((5))
    net = tf.keras.layers.Dense(4)(inpt)
    mod2 = tf.keras.Model(inpt, net)
    out2 = mod2(test_inpt).numpy()
    
    # updating model with itself should give rolling sum = 0
    rollingsum = exponential_model_update(mod1, mod1).numpy()
    assert rollingsum == 0.
    
    # setting exponential parameter to 1 should return original model
    rs = exponential_model_update(mod1, mod2, 1.)
    out3 = mod1(test_inpt).numpy()
    assert np.sum((out3 - out1)**2) < 1e-5
    
    # setting exponential parameter to 0 should return second model
    rs = exponential_model_update(mod1, mod2, 0.)
    out4 = mod1(test_inpt).numpy()
    assert np.sum((out4 - out2)**2) < 1e-5
    
    