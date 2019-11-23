# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf

from patchwork.feature import _contextencoder as ce


def test_context_encoder_mask_generator_outputs():
    maskgen = ce.mask_generator(32,32,3)
    mask = next(maskgen)
    assert isinstance(mask, np.ndarray)
    assert mask.shape == (32,32,3)
    assert mask.dtype == np.float32


def test_context_encoder_make_test_mask_generator_outputs():
    mask = ce._make_test_mask(32,32,3)
    assert isinstance(mask, np.ndarray)
    assert mask.shape == (32,32,3)
    assert mask.dtype == np.float32
    
    
def test_build_context_encoder_dataset(test_png_path):
    ds = ce._build_context_encoder_dataset([test_png_path],
                                                input_shape=(64,64,3),
                                                shuffle=True, 
                                                num_parallel_calls=1,
                                                batch_size=1, prefetch=False)
    for record in ds:
        pass
    
    assert len(record) == 2
    x = record[0].numpy()
    y = record[1].numpy()
    
    assert isinstance(x, np.ndarray)
    assert x.shape == (1,64,64,3)
    assert x.dtype == np.float32

    assert isinstance(y, np.ndarray)
    assert y.shape == (1,64,64,3)
    assert y.dtype == np.float32
    
    
def test_build_test_dataset(test_png_path):
    ds = ce._build_test_dataset([test_png_path], 
                                input_shape=(64,64,3))
    
    assert len(ds) == 2
    x = ds[0]
    y = ds[1]
    
    assert isinstance(x, np.ndarray)
    assert x.shape == (1,64,64,3)
    assert x.dtype == np.float32

    assert isinstance(y, np.ndarray)
    assert y.shape == (1,64,64,3)
    assert y.dtype == np.float32
    
    
    
def test_stabilize():
    assert ce._stabilize(0.5).numpy() == 0.5
    assert ce._stabilize(0.).numpy() > 0
    assert ce._stabilize(1.).numpy() < 1
    
    
# MOCK KERAS MODELS FOR TESTING UPDATE FUNCTIONS 
opt = tf.keras.optimizers.SGD(1e-3)
# INPAINTER
inpt = tf.keras.layers.Input((None,None,3))
outpt = tf.keras.layers.Conv2D(3, 1)(inpt)
mock_inpainter = tf.keras.Model(inpt, outpt)
# DISCRIMINATOR
inpt = tf.keras.layers.Input((None,None,3))
pooled = tf.keras.layers.GlobalMaxPool2D()(inpt)
outpt = tf.keras.layers.Dense(1, activation="sigmoid")(pooled)
mock_discriminator = tf.keras.Model(inpt, outpt)

    
def test_context_encoder_inpainter_training_step(test_png_path):
    x, y = ce._build_test_dataset([test_png_path], input_shape=(64,64,3))
    losses = ce.inpainter_training_step(opt, 
                                        mock_inpainter, 
                                        mock_discriminator, x, y)
    assert len(losses) == 3
    for i in range(3):
        assert losses[i].numpy().size == 1
        assert losses[i].dtype == tf.float32


def test_context_encoder_discriminator_training_step(test_png_path):
    x, y = ce._build_test_dataset([test_png_path], input_shape=(64,64,3))
    loss = ce.discriminator_training_step(opt, 
                                          mock_inpainter, 
                                          mock_discriminator, x, y)
    assert loss.numpy().size == 1
    assert loss.dtype == tf.float32
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    