# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf

from patchwork.feature.clip import build_image_encoder
from patchwork.feature.clip import compute_nce_loss
from patchwork.feature.clip import save_clip_dataset_to_tfrecords
from patchwork.feature.clip import load_clip_dataset_from_tfrecords
    
class MockEncoder():
    def __init__(self, maxlen):
        self._maxlen = maxlen
        
    def encode(self, x, **kwargs):
        return np.arange(self._maxlen)
    
def test_build_image_encoder():
    inpt = tf.keras.layers.Input((None, None, 3))
    x = tf.keras.layers.Conv2D(3,1)(inpt)
    fcn = tf.keras.Model(inpt, x)
    encoder = build_image_encoder(fcn, output_dim=7)
    assert isinstance(encoder, tf.keras.Model)
    assert encoder.output_shape == (None, 7)
    
    
def test_save_and_load_tfrecord(test_png_path, tmp_path_factory):
    N = 10
    maxlen = 13
    imshape = (32,32)
    num_channels = 3
    num_parallel_calls = 1
    
    imfiles = [test_png_path]*N
    labels = ["somebody once told me, the world is gonna roll me"]*N
    
    encoder = MockEncoder(maxlen)
    
    # SAVE IT TO TFRECORD FILES
    fn = str(tmp_path_factory.mktemp("data"))
    save_clip_dataset_to_tfrecords(imfiles, labels, encoder, fn,
                                   num_shards=2, maxlen=maxlen, 
                                   imshape=imshape, num_channels=num_channels,
                                   num_parallel_calls=num_parallel_calls)
    
    ds = load_clip_dataset_from_tfrecords(fn, imshape, num_channels,
                                          shuffle=False, maxlen=maxlen)
    
    for x,y in ds:
        break
    
    assert isinstance(ds, tf.data.Dataset)
    assert x.shape == (32,32,3)
    assert y.shape == (maxlen)

    
"""
def test_compute_nce_loss():
    N = 7
    d = 13
    img_embed = np.random.normal(0, 1, (N,d)).astype(np.float32)
    text_embed = np.random.normal(0,1, (N,d)).astype(np.float32)
    loss = compute_nce_loss(img_embed, text_embed).numpy()
    assert loss > 0
    """
