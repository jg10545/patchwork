# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf

from patchwork.feature._clip import build_image_encoder
from patchwork.feature._clip import compute_nce_loss
    

    
def test_build_image_encoder():
    inpt = tf.keras.layers.Input((None, None, 3))
    x = tf.keras.layers.Conv2D(3,1)(inpt)
    fcn = tf.keras.Model(inpt, x)
    encoder = build_image_encoder(fcn, output_dim=7)
    assert isinstance(encoder, tf.keras.Model)
    assert encoder.output_shape == (None, 7)
    
    
"""
def test_compute_nce_loss():
    N = 7
    d = 13
    img_embed = np.random.normal(0, 1, (N,d)).astype(np.float32)
    text_embed = np.random.normal(0,1, (N,d)).astype(np.float32)
    loss = compute_nce_loss(img_embed, text_embed).numpy()
    assert loss > 0
    """
