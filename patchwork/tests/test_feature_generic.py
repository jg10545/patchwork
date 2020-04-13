# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf

from patchwork.feature._generic import linear_classification_test


class MockLabelDict():
    """
    something that'll act like a dictionary for linear_classification_test
    without me needing to have a bunch of fixture images
    """
    def __init__(self, a, b, c=None):
        if c is None:
            self._keys = 10*[a] + 10*[b]
        else:
            self._keys = 10*[",".join([a,c])] + 10*[",".join([b,c])]
        self._values = 10*["foo"] + 10*["bar"]
        
    def keys(self):
        return self._keys
    
    def values(self):
        return self._values
    
    def __len__(self):
        return 20


inpt = tf.keras.layers.Input((None, None, 3))
conv = tf.keras.layers.Conv2D(5, 1, (2,2))(inpt)
fcn = tf.keras.Model(inpt, conv)


inpt2 = tf.keras.layers.Input((None, None, 1))
conv2 = tf.keras.layers.Conv2D(5, 1, (2,2))(inpt)
concat = tf.keras.layers.Concatenate()([conv, conv2])
multi_input_fcn = tf.keras.Model([inpt, inpt2], concat)


def test_linear_classification_test(test_png_path, test_jpg_path):
    labeldict = MockLabelDict(test_png_path, test_jpg_path)
    
    acc,cm = linear_classification_test(fcn, labeldict, 
                                        imshape=(20,20),
                                        num_channels=3,
                                        norm=255,
                                        single_channel=False,
                                        batch_size=10)

    assert isinstance(acc, float)
    assert acc <= 1
    assert acc >= 0
    assert isinstance(cm, np.ndarray)
    
    

def test_linear_classification_test_dual_input(test_png_path, test_jpg_path,
                                               test_single_channel_png_path):
    labeldict = MockLabelDict(test_png_path, test_jpg_path,
                              test_single_channel_png_path)
    
    acc,cm = linear_classification_test(multi_input_fcn, labeldict, 
                                        imshape=[(20,20),(20,20)],
                                        num_channels=[3,1],
                                        norm=255,
                                        single_channel=[False,True],
                                        batch_size=10)

    assert isinstance(acc, float)
    assert acc <= 1
    assert acc >= 0
    assert isinstance(cm, np.ndarray)