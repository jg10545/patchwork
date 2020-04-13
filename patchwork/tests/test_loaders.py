# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
from patchwork.loaders import _image_file_dataset, dataset, stratified_training_dataset
from patchwork.loaders import _build_load_function



def test_build_load_function_png(test_png_path):
    load_fn = _build_load_function((64,64), 255, 3, False)
    loaded = load_fn(test_png_path,0)
    
    assert isinstance(loaded, tf.Tensor)
    assert loaded.numpy().shape == (64,64,3)
    assert loaded.numpy().max() <= 1



def test_build_load_function_jpg(test_jpg_path):
    load_fn = _build_load_function((17,23), 255, 3, False)
    loaded = load_fn(test_jpg_path,1)
    
    assert isinstance(loaded, tf.Tensor)
    assert loaded.numpy().shape == (17,23,3)
    assert loaded.numpy().max() <= 1



def test_build_load_function_single_channel_png(test_single_channel_png_path):
    load_fn = _build_load_function((13,17), 255, 3, True)
    loaded = load_fn(test_single_channel_png_path,0)
    
    assert isinstance(loaded, tf.Tensor)
    assert loaded.numpy().shape == (13,17,3)
    assert loaded.numpy().max() <= 1

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


def test_image_file_dataset_multi_input(test_png_path):
    imfiles = [[test_png_path], [test_png_path]]
    ds = _image_file_dataset(imfiles, 
                             imshape=[(33,21), (17,23)],
                             norm=[255,255], 
                             num_channels=[3,3],
                             single_channel=[False,False],
                             shuffle=False)
    for x,y in ds:
        x = x.numpy()
        y = y.numpy()
    
    assert isinstance(ds, tf.data.Dataset)
    assert isinstance(x, np.ndarray)
    assert x.max() <= 1
    assert x.min() >= 0
    assert x.shape == (33, 21, 3)
    assert y.shape == (17, 23, 3)


def test_image_file_dataset_multi_input_and_labels(test_png_path):
    imfiles = [2*[test_png_path], 2*[test_png_path]]
    ys = np.zeros(2, dtype=np.int64)
    ds = _image_file_dataset(imfiles, ys=ys,
                             imshape=[(33,21), (17,23)],
                             norm=[255,255], 
                             num_channels=[3,3],
                             single_channel=[False,False],
                             shuffle=False)
    for (x0, x1), y in ds:
        x0 = x0.numpy()
        x1 = x1.numpy()
        y = y.numpy()
        break
    
    assert isinstance(ds, tf.data.Dataset)
    assert isinstance(x0, np.ndarray)
    assert x0.max() <= 1
    assert x0.min() >= 0
    assert x0.shape == (33, 21, 3)
    assert x1.shape == (17, 23, 3)
    assert y.size == 1


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
                     batch_size=5, augment={"flip_left_right":True})
    
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
    
    
def test_dual_input_dataset_without_augmentation(test_png_path,
                                                 test_single_channel_png_path):
    imfiles = [[test_png_path]*10, [test_single_channel_png_path]*10]
    
    ds, ns = dataset(imfiles, ys=None, 
                     imshape=[(11,17), (19,23)],
                     num_channels=[3,1], norm=[255,255],
                     single_channel=[False, False],
                     batch_size=5, augment=False)
    
    for x,y in ds:
        x = x.numpy()
        y = y.numpy()
        break
    
    assert isinstance(ds, tf.data.Dataset)
    assert ns == 2
    assert x.shape == (5, 11, 17, 3)
    assert y.shape == (5, 19, 23, 1)
    
def test_dual_input_dataset_with_labels(test_png_path,
                                                 test_single_channel_png_path):
    imfiles = [[test_png_path]*10, [test_single_channel_png_path]*10]
    ys = np.zeros(10, dtype=np.int64)
    
    ds, ns = dataset(imfiles, ys=ys, 
                     imshape=[(11,17), (19,23)],
                     num_channels=[3,1], norm=[255,255],
                     single_channel=[False, False],
                     batch_size=5, augment=False)
    
    for (x0,x1),y in ds:
        x0 = x0.numpy()
        x1 = x1.numpy()
        y = y.numpy()
        break
    
    assert isinstance(ds, tf.data.Dataset)
    assert ns == 2
    assert x0.shape == (5, 11, 17, 3)
    assert x1.shape == (5, 19, 23, 1)
    assert y.shape == (5,)
    assert y.dtype == np.int64
    
def test_dual_input_dataset_with_augmentation(test_png_path,
                                                 test_single_channel_png_path):
    imfiles = [[test_png_path]*10, [test_single_channel_png_path]*10]
    aug = {"flip_left_right":True}
    
    ds, ns = dataset(imfiles, ys=None, 
                     imshape=(11,17),
                     num_channels=[3,1], norm=255,
                     #single_channel=[False, False],
                     single_channel=[False, True],
                     batch_size=5, augment=aug)
    
    for x,y in ds:
        x = x.numpy()
        y = y.numpy()
        break
    
    assert isinstance(ds, tf.data.Dataset)
    assert ns == 2
    assert x.shape == (5, 11, 17, 3)
    assert y.shape == (5, 11, 17, 1)
    
def test_stratified_training_dataset(test_png_path):
    imfiles = [test_png_path]*10
    labels = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    
    ds = stratified_training_dataset(imfiles,
                                         labels, imshape=(13,11),
                                         num_channels=3,
                                         batch_size=10,
                                         mult=1,
                                         augment={"flip_up_down":True})
    
    for x,y in ds:
        x = x.numpy()
        y = y.numpy()
        
    #assert ns == 1
    # check to make sure it stratified correctly
    assert y.mean() == 0.5
    
    

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
    
    