# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf

from patchwork._augment import augment_function


test_shape = (64,64,3)
test_img = np.zeros(test_shape, dtype=np.float32)
test_img_tensor = tf.constant(test_img, dtype=tf.float32)


def test_default_augment():
    augfunc = augment_function(test_shape[:2])
    augmented = augfunc(test_img_tensor)
    
    assert isinstance(augmented, tf.Tensor)
    assert augmented.get_shape() == test_img_tensor.get_shape()