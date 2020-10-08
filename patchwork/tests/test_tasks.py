# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf

from patchwork._tasks import _rotate, _build_rotation_dataset

    
def test_build_rotation_dataset(test_png_path):
    filepaths = 10*[test_png_path]
    ds = _build_rotation_dataset(filepaths, imshape=(32,32),
                                     batch_size=5, 
                                     augment={"flip_left_right":True})
    assert isinstance(ds, tf.data.Dataset)
    for x,y in ds:
        x = x.numpy()
        y = y.numpy()
        break
    
    assert x.shape == (5,32,32,3)
    assert y.shape == (5,)
    assert y.max() < 4
    assert y.min() >= 0
    
