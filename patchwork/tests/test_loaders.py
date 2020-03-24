# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
from patchwork.loaders import _image_file_dataset, dataset, stratified_training_dataset
from patchwork.loaders import _sobelize



def test_image_file_dataset(test_png_path):
    imfiles = [test_png_path]
    ds = _image_file_dataset(imfiles, imshape=(33,21),
                             norm=255, num_channels=3,
                             shuffle=False)
    for x in ds:
        x = x.numpy()
    
    assert isinstance(ds, tf.data.Dataset)
    assert isinstance(x, np.ndarray)
    assert x.max() <= 1
    assert x.min() >= 0
    assert x.shape == (33, 21, 3)



def test_dataset_without_augmentation(test_png_path):
    imfiles = [test_png_path]*10
    
    ds, ns = dataset(imfiles, ys=None, imshape=(11,17),
                     num_channels=3, norm=255,
                     batch_size=5, augment=False)
    
    for x in ds:
        x = x.numpy()
        break
    
    assert isinstance(ds, tf.data.Dataset)
    assert ns == 2
    assert x.shape == (5, 11, 17, 3)
    
    
def test_dataset_with_augmentation(test_png_path):
    imfiles = [test_png_path]*10
    
    ds, ns = dataset(imfiles, ys=None, imshape=(11,17),
                     num_channels=3, norm=255,
                     batch_size=5, augment={})
    
    for x in ds:
        x = x.numpy()
        break
    
    assert isinstance(ds, tf.data.Dataset)
    assert ns == 2
    assert x.shape == (5, 11, 17, 3)
    
    
def test_dataset_with_labels(test_png_path):
    imfiles = [test_png_path]*10
    labels = np.arange(10)
    
    ds, ns = dataset(imfiles, ys=labels, imshape=(11,17),
                     num_channels=3, norm=255,
                     batch_size=5, augment=False)
    
    for x,y in ds:
        x = x.numpy()
        y = y.numpy()
        break
    
    assert (y == np.arange(5)).all()
    
    
def test_stratified_training_dataset(test_png_path):
    imfiles = [test_png_path]*10
    labels = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    
    ds, ns = stratified_training_dataset(imfiles,
                                         labels, imshape=(13,11),
                                         num_channels=3,
                                         batch_size=10,
                                         mult=1,
                                         augment={})
    
    for x,y in ds:
        x = x.numpy()
        y = y.numpy()
        
    assert ns == 1
    # check to make sure it stratified correctly
    assert y.mean() == 0.5
    
    
def test_sobelize():
    inpt = tf.zeros((7,11,4), dtype=tf.float32)
    outpt = _sobelize(inpt)
    
    assert outpt.shape == (7,11,3)
    assert outpt.numpy()[:,:,2].max() == 0
    

def test_image_file_dataset_stripping_alpha_channel(test_rgba_png_path, test_png_path):
    imfiles = [test_rgba_png_path, test_png_path]
    ds = _image_file_dataset(imfiles, imshape=(31,23),
                             norm=255, num_channels=3,
                             shuffle=False)
    for x in ds:
        x = x.numpy()
    
        assert isinstance(ds, tf.data.Dataset)
        assert isinstance(x, np.ndarray)
        assert x.max() <= 1
        assert x.min() >= 0
        assert x.shape == (31, 23, 3)

    
    
def test_dataset_works_with_keras_api(test_png_path):
    """
    Make sure we can use pw.loaders.dataset() with
    the high-level keras API
    """
    imfiles = [test_png_path]*10
    labels = np.random.randint(0,2, size=10)
    
    ds, ns = dataset(imfiles, ys=labels, imshape=(11,17),
                     num_channels=3, norm=255,
                     batch_size=5, augment=False)
    
    inpt = tf.keras.layers.Input((None, None,3))
    net = tf.keras.layers.Conv2D(2, 1)(inpt)
    net = tf.keras.layers.GlobalMaxPool2D()(net)
    net = tf.keras.layers.Dense(2, activation="softmax")(net)
    model = tf.keras.Model(inpt, net)
    
    model.compile("SGD", loss=tf.keras.losses.sparse_categorical_crossentropy)
    
    hist = model.fit(ds, steps_per_epoch=ns, epochs=1)
    assert isinstance(hist, tf.keras.callbacks.History)
    print(hist)
    
    