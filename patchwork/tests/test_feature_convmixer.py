# -*- coding: utf-8 -*-
import tensorflow as tf

from patchwork.feature._convmixer import build_convmixer_fcn


def test_build_convmixer_fcn():
    dim = 7
    depths = [2,3,5]
    for d in depths:
        model = build_convmixer_fcn(dim, d, input_shape=(128,128,3))
        assert isinstance(model, tf.keras.Model)
        assert model.output_shape == (None, 18, 18, dim)
        
        
def test_build_convmixer_fcn_with_relu():
    dim = 7
    depths = [2,3,5]
    for d in depths:
        model = build_convmixer_fcn(dim, d, input_shape=(128,128,3), gelu=False)
        assert isinstance(model, tf.keras.Model)
        assert model.output_shape == (None, 18, 18, dim)

