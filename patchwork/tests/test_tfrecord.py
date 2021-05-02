# -*- coding: utf-8 -*-
import tensorflow as tf
from patchwork._tfrecord import save_dataset_to_tfrecords
from patchwork.loaders import load_dataset_from_tfrecords




def test_save_and_load_tfrecord(test_png_path, tmp_path_factory):
    imfiles = [test_png_path]*10
    
    # SAVE IT TO TFRECORD FILES
    fn = str(tmp_path_factory.mktemp("data"))
    save_dataset_to_tfrecords(imfiles, fn, num_shards=2,
                              imshape=(32,32), num_channels=3,
                              norm=255)
    
    ds = load_dataset_from_tfrecords(fn)
    for x in ds:
        break
    
    assert isinstance(ds, tf.data.Dataset)
    assert x.shape == (32,32,3)
