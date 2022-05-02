# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf

from patchwork.feature._vicreg import _build_expander, _cov_loss

inpt = tf.keras.layers.Input((None, None, 3))
net = tf.keras.layers.Conv2D(3,1)(inpt)
fcn = tf.keras.Model(inpt, net)


def test_build_expander():
    num_hidden = 17
    expander = _build_expander(fcn, (5,7), 3, num_hidden)
    
    assert isinstance(expander, tf.keras.Model)
    assert expander.output_shape == (None, num_hidden)
    
    
    
    
    
def test_cov_loss():
    x = np.zeros((7,5)).astype(np.float32)
    cov = _cov_loss(x)
    assert cov.numpy() == 0