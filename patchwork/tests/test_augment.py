# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf

from patchwork._augment import _random_rotate, _random_crop, _random_distort, _augment

test_shape = (64,64,3)
test_img = np.zeros(test_shape, dtype=np.float32)
test_img_tensor = tf.constant(test_img, dtype=tf.float32)


def test_random_rotate():
    rotated = _random_rotate(test_img_tensor)
    with tf.Session() as sess:
        rotated_computed = sess.run(rotated)
        
    assert isinstance(rotated, tf.Tensor)
    assert rotated_computed.shape == test_shape
    
    
def test_random_crop():
    cropped = _random_crop(test_img_tensor)
    with tf.Session() as sess:
        cropped_computed = sess.run(cropped)
        
    assert isinstance(cropped, tf.Tensor)
    assert cropped_computed.shape == test_shape
    
    
def test_random_distort():
    distorted = _random_distort(test_img_tensor)
    with tf.Session() as sess:
        distorted_computed = sess.run(distorted)
        
    assert isinstance(distorted, tf.Tensor)
    assert distorted_computed.shape == test_shape
    
def test_random_augment():
    augmented = _augment(test_img_tensor)
    with tf.Session() as sess:
        augmented_computed = sess.run(augmented)
        
    assert isinstance(augmented, tf.Tensor)
    assert augmented_computed.shape == test_shape