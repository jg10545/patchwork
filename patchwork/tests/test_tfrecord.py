# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
tf.random.set_seed(1)

from patchwork._tfrecord import save_dataset_to_tfrecords
from patchwork.loaders import load_dataset_from_tfrecords




def test_save_and_load_tfrecord(test_png_path, tmp_path_factory):
    imfiles = [test_png_path]*10
    
    # SAVE IT TO TFRECORD FILES
    fn = str(tmp_path_factory.mktemp("data"))
    save_dataset_to_tfrecords(imfiles, fn, num_shards=2,
                              imshape=(32,32), num_channels=3,
                              norm=255)
    
    ds = load_dataset_from_tfrecords(fn, (32,32), 3)
    for x in ds:
        break
    
    assert isinstance(ds, tf.data.Dataset)
    assert x.shape == (32,32,3)



def test_save_and_load_tfrecord_from_dataset(test_png_path, tmp_path_factory):
    data = np.zeros((10,32,32,3),dtype=np.float32)
    ds = tf.data.Dataset.from_tensor_slices(data)
    
    # SAVE IT TO TFRECORD FILES
    fn = str(tmp_path_factory.mktemp("data2"))
    save_dataset_to_tfrecords(ds, fn, num_shards=2,
                              imshape=(32,32), num_channels=3,
                              norm=255)
    
    ds = load_dataset_from_tfrecords(fn, (32,32), 3)
    for x in ds:
        break
    
    assert isinstance(ds, tf.data.Dataset)
    assert x.shape == (32,32,3)


def test_save_and_load_tfrecord_from_pair_dataset(test_png_path, tmp_path_factory):
    # make sure the tfrecord writer works if the user wants to save pairs
    # of image (for use with custom sampling methods in SimCLR, MoCo, and HCL)
    data1 = np.zeros((10,32,32,3),dtype=np.float32)
    data2 = np.zeros((10,32,32,3),dtype=np.float32)
    ds = tf.data.Dataset.from_tensor_slices((data1, data2))
    print(ds)
    
    # SAVE IT TO TFRECORD FILES
    fn = str(tmp_path_factory.mktemp("data3"))
    save_dataset_to_tfrecords(ds, fn, num_shards=2,
                              imshape=(32,32), num_channels=3,
                              norm=255)
    
    ds = load_dataset_from_tfrecords(fn, (32,32), 3, num_images=2)
    for x in ds:
        break
    
    assert isinstance(ds, tf.data.Dataset)
    print(x)
    assert len(x) == 2
    assert x[0].shape == (32,32,3)
    assert x[1].shape == (32,32,3)
