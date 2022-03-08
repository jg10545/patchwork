# -*- coding: utf-8 -*-

import tensorflow as tf

from patchwork._convnext import _add_convnext_block


def test_add_convnext_block():
    inpt = tf.keras.layers.Input((11, 13, 7))
    outpt = _add_convnext_block(inpt)

    assert inpt.shape.as_list() == outpt.shape.as_list()