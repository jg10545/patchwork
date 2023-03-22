# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf

tf.random.seed(1)

from patchwork.feature._generic import linear_classification_test
from patchwork.feature._generic import build_optimizer
from patchwork.feature._generic import GenericExtractor
from patchwork.loaders import dataset


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
                                        batch_size=1)

    assert isinstance(acc, float)
    assert acc <= 1
    assert acc >= 0
    assert isinstance(cm, np.ndarray)
    
    
    
    
def test_linear_classification_test_dataset_input(test_png_path, test_jpg_path):
    filepaths = [test_png_path, test_jpg_path]*5
    ys = [0,1]*5
    ds = dataset(filepaths, ys=ys, imshape=(20,20), batch_size=2)[0]
    
    acc,cm = linear_classification_test(fcn, ds)

    assert isinstance(acc, float)
    assert acc <= 1
    assert acc >= 0
    assert isinstance(cm, np.ndarray)
    
    
def test_build_optimizer_constant_rate():
    lr = 0.01
    opt = build_optimizer(lr, lr_decay=0)
    assert hasattr(opt, "lr")
    # floating point rounding issues
    assert abs(opt.lr.numpy() - lr) < 1e-8
    
    
def test_build_optimizer_exponential_decay():
    lr = 0.01
    lr_decay = 100
    decay_type = "exponential"
    opt = build_optimizer(lr, lr_decay=lr_decay,
                          decay_type=decay_type)
    assert hasattr(opt, "lr")
    # floating point rounding issues
    assert abs(opt.lr(lr_decay).numpy() - lr/2) < 1e-8
    
def test_build_optimizer_cosine_decay():
    lr = 0.01
    lr_decay = 1000
    decay_type = "cosine"
    opt = build_optimizer(lr, lr_decay=lr_decay,
                          decay_type=decay_type)
    assert hasattr(opt, "lr")
    assert opt.lr(lr_decay-1).numpy() < 1e-3
    assert opt.lr(lr_decay).numpy() == opt.lr(0).numpy()
    
    

ds = tf.data.Dataset.from_tensor_slices(np.zeros((10,1), dtype=np.float32))

strat = tf.distribute.OneDeviceStrategy("/cpu:0")
extractor = GenericExtractor()
extractor_with_strat = GenericExtractor(strategy=strat)


def _mock_train_step(x,y):
    return {"foo":x}

def test_genericextractor_without_distribution():
    
    with extractor.scope():
        inpt = tf.keras.layers.Input((1,))
        outpt = tf.keras.layers.Dense(1)(inpt)
        model = tf.keras.Model(inpt, outpt)
        
    dataset = extractor._distribute_dataset(ds)
    step = extractor._distribute_training_function(_mock_train_step)
    
    for x in dataset:
        output = step(x,x)
        
    assert isinstance(output, dict)
    assert "foo" in output
    assert output["foo"].numpy() == 0.


def test_genericextractor_with_distribution():
    
    with extractor_with_strat.scope():
        inpt = tf.keras.layers.Input((1,))
        outpt = tf.keras.layers.Dense(1)(inpt)
        model = tf.keras.Model(inpt, outpt)
        
    dataset = extractor_with_strat._distribute_dataset(ds)
    step = extractor_with_strat._distribute_training_function(_mock_train_step)
    
    for x in dataset:
        output = step(x,x)
        
    assert isinstance(output, dict)
    assert "foo" in output
    assert output["foo"].numpy() == 0.
