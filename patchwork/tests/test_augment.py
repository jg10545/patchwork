# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf

from patchwork._augment import _random_rotate, _random_crop, _random_distort, _augment

test_shape = (64,64,3)
test_img = np.zeros(test_shape, dtype=np.float32)
test_img_tensor = tf.constant(test_img, dtype=tf.float32)


def test_random_rotate():
    rotated = _random_rotate(test_img_tensor)
        
    assert isinstance(rotated, tf.Tensor)
    assert rotated.numpy().shape == test_shape
    
    
def test_random_crop():
    cropped = _random_crop(test_img_tensor)
        
    assert isinstance(cropped, tf.Tensor)
    assert cropped.numpy().shape == test_shape
    
    
def test_random_distort():
    distorted = _random_distort(test_img_tensor)
        
    assert isinstance(distorted, tf.Tensor)
    assert distorted.numpy().shape == test_shape
    
def test_random_augment():
    augmented = _augment(test_img_tensor)
        
    assert isinstance(augmented, tf.Tensor)
    assert augmented.numpy().shape == test_shape