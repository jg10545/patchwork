# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf

from patchwork.viz._integrated_gradients import _interpolate_images
from patchwork.viz._integrated_gradients import _compute_gradients
from patchwork.viz._integrated_gradients import _integral_approximation


def test_interpolate_images():
    image = np.random.uniform(0, 1, size=(32,32,3)).astype(np.float32)
    base = tf.zeros_like(image)
    alphas = tf.linspace(0., 1., 5)
    
    interpolated = _interpolate_images(base, image, alphas)
    assert interpolated.shape == (5,32,32,3)
    
    
def test_compute_gradients():
    images = tf.constant(np.random.uniform(0, 1, size=(5,7,11,3)).astype(np.float32))
    aug_embeds = np.random.normal(0, 1, (13,3)).astype(np.float32)
    
    inpt = tf.keras.layers.Input((None, None, 3))
    outpt = tf.keras.layers.GlobalAvgPool2D()(inpt)
    model = tf.keras.Model(inpt, outpt)
    
    grads = _compute_gradients(images, model, aug_embeds)
    assert grads.shape == images.shape
    
def test_compute_gradients_with_negatives():
    images = tf.constant(np.random.uniform(0, 1, size=(5,7,11,3)).astype(np.float32))
    aug_embeds = np.random.normal(0, 1, (13,3)).astype(np.float32)
    neg_embeds = np.random.normal(0, 1, (17,3)).astype(np.float32)
    
    inpt = tf.keras.layers.Input((None, None, 3))
    outpt = tf.keras.layers.GlobalAvgPool2D()(inpt)
    model = tf.keras.Model(inpt, outpt)
    
    grads = _compute_gradients(images, model, aug_embeds, neg_embeds)
    assert grads.shape == images.shape
    
def test_integral_approximation():
    grads = tf.constant(np.random.normal(0, 1, size=(5, 7, 11, 3)).astype(np.float32))
    approx = _integral_approximation(grads)
    
    assert approx.shape == grads.shape.as_list()[1:]