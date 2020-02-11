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


def test_exponential_model_update():
    fcn = tf.keras.applications.VGG16(weights=None, include_top=False)
    assert exponential_model_update(fcn, fcn).numpy() == 0.