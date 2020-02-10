# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf

from patchwork._augment import augment_function, _poisson, _random_zoom, _choose


test_shape = (64,64,3)
test_img = np.zeros(test_shape, dtype=np.float32)
test_img_tensor = tf.constant(test_img, dtype=tf.float32)


def test_default_augment():
    augfunc = augment_function(test_shape[:2])
    augmented = augfunc(test_img_tensor)
    
    assert isinstance(augmented, tf.Tensor)
    assert augmented.get_shape() == test_img_tensor.get_shape()
    
    
def test_poisson():
    c = _poisson(100)
    
    assert c > 0
    assert c.dtype == tf.int32
    
    
def test_random_zoom():
    img = np.random.uniform(0,1, (32,32,3)).astype(np.float32)
    
    zoomed = _random_zoom(img, (32,32))
    assert img.shape == zoomed.shape
    assert (zoomed.numpy() == img).all() == False
    
def test_choose():
    choice = _choose(0.5)
    assert choice.dtype == tf.bool