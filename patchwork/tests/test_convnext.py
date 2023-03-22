# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np

from patchwork._convnext import _add_convnext_block, build_convnext_fcn


def test_build_convnext():
    """
    checks that we get the expected output shape
    """
    x = tf.ones((1, 224, 224, 3))
    fcn = build_convnext_fcn("T", use_grn=True, num_channels=3)
    print(fcn(x).shape)
    assert fcn(x).shape == (1, 7, 7, 768)


def test_convnext_grn():
    """
    runs the same input with and without GRN and checks that
    the result is the same
    """
    tf.config.experimental.enable_op_determinism()
    x = tf.random.uniform((1, 224, 224, 1))
    tf.keras.utils.set_random_seed(1)
    fcn_nogrn = build_convnext_fcn("T", use_grn=False, num_channels=1)
    tf.keras.utils.set_random_seed(1)
    fcn_grn = build_convnext_fcn("T", use_grn=True, num_channels=1)
    nogrn_out = fcn_nogrn(x)
    grn_out = fcn_grn(x)
    assert np.array_equal(nogrn_out.numpy(), grn_out.numpy())


def test_add_convnext_block():
    inpt = tf.keras.layers.Input((11, 13, 7))
    outpt = _add_convnext_block(inpt)

    assert inpt.shape.as_list() == outpt.shape.as_list()
