# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf

from patchwork._util import shannon_entropy, tiff_to_array, _load_img
from patchwork._util import compute_l2_loss, _compute_alignment_and_uniformity




def test_shannon_entropy_agg_sum():
    maxent = np.array([[0.5,0.5],[0.5,0.5]])
    assert shannon_entropy(maxent, how="sum")[0] == 2.#1.0
    
def test_shannon_entropy_agg_max():
    maxent = np.array([[0.5,0.5],[0.5,0.5]])
    assert shannon_entropy(maxent, how="max")[0] == 1.#1.0
    

    
"""    
def test_tiff_to_array(test_tif_path):
    img_arr = tiff_to_array(test_tif_path, num_channels=-1)
    
    assert isinstance(img_arr, np.ndarray)
    assert len(img_arr.shape) == 3
    assert img_arr.shape[2] == 3
    
    
def test_tiff_to_array_fixed_channels(test_tif_path):
    img_arr = tiff_to_array(test_tif_path, num_channels=2)
    
    assert isinstance(img_arr, np.ndarray)
    assert len(img_arr.shape) == 3
    assert img_arr.shape[2] == 2
    
    
def test_geotiff_to_array_fixed_channels(test_geotif_path):
    img_arr = tiff_to_array(test_geotif_path, num_channels=4, norm=1)
    
    assert isinstance(img_arr, np.ndarray)
    assert len(img_arr.shape) == 3
    assert img_arr.shape[2] == 4
"""
    
    
def test_load_img_on_png(test_png_path):
    img_arr = _load_img(test_png_path)
    assert isinstance(img_arr, np.ndarray)
    assert len(img_arr.shape) == 3
    assert img_arr.shape[2] == 3


def test_load_img_on_jpg(test_jpg_path):
    img_arr = _load_img(test_jpg_path)
    assert isinstance(img_arr, np.ndarray)
    assert len(img_arr.shape) == 3
    assert img_arr.shape[2] == 3

def test_load_img_on_png_with_resize(test_png_path):
    img_arr = _load_img(test_png_path, resize=(71,71))
    assert isinstance(img_arr, np.ndarray)
    assert len(img_arr.shape) == 3
    assert img_arr.shape == (71,71,3)    
    
"""
def test_load_img_on_geotif(test_geotif_path):
    img_arr = _load_img(test_geotif_path, num_channels=4, norm=1)
    assert isinstance(img_arr, np.ndarray)
    assert len(img_arr.shape) == 3
    assert img_arr.shape[2] == 4
"""

def test_load_single_channel_img(test_single_channel_png_path):
    img_arr = _load_img(test_single_channel_png_path, resize=(32,32),
                        num_channels=1)
    assert isinstance(img_arr, np.ndarray)
    assert len(img_arr.shape) == 3
    assert img_arr.shape[2] == 1


def test_load_and_stack_single_channel_img(test_single_channel_png_path):
    img_arr = _load_img(test_single_channel_png_path, num_channels=3)
    assert isinstance(img_arr, np.ndarray)
    assert len(img_arr.shape) == 3
    assert img_arr.shape[2] == 3
    
    
def test_l2_loss():
    # build a simple model
    inpt = tf.keras.layers.Input(3)
    net = tf.keras.layers.Dense(5)(inpt)
    model = tf.keras.Model(inpt, net)
    # check the L2 loss; should be a scalar bigger than zero
    assert compute_l2_loss(model).numpy() > 0
    # should also work with multiple inputs
    assert compute_l2_loss(model, model).numpy() > 0
    # set all trainable weights to zro
    new_weights = [np.zeros(x.shape, dtype=np.float32)
                   for x in model.get_weights()]
    model.set_weights(new_weights)
    assert compute_l2_loss(model).numpy() == 0.
    # add a batch norm layer- shouldn't change anything
    # because the function should skip that layer
    net = tf.keras.layers.BatchNormalization()(net)
    model = tf.keras.Model(inpt,net)
    assert compute_l2_loss(model).numpy() == 0.
    
    
def test_compute_alignment_and_uniformity():
    X = np.random.normal(0, 1, size=(32, 5,5,3)).astype(np.float32)
    ds = tf.data.Dataset.from_tensor_slices((X,X))
    ds = ds.batch(8)

    inpt = tf.keras.layers.Input((None, None, 3))
    net = tf.keras.layers.Conv2D(7,1)(inpt)
    mod = tf.keras.Model(inpt,net)
    
    al, un = _compute_alignment_and_uniformity(ds, mod)
    
    assert al >= 0
    assert un <= 0