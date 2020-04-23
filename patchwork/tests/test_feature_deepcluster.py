# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf

from patchwork.feature._deepcluster import cluster, _build_model, build_deepcluster_training_step

inpt = tf.keras.layers.Input((None, None, 3))
conv = tf.keras.layers.Conv2D(2, 1)(inpt)
fcn = tf.keras.Model(inpt, conv)

net = tf.keras.layers.GlobalAveragePooling2D()(conv)
dense = tf.keras.layers.Dense(7)(net)
model = tf.keras.Model(inpt, dense)


def test_cluster():
    vecs = np.random.uniform(0, 1, size=(37,23))
    testvecs = np.random.uniform(0, 1, size=(5,23))
    
    predictions, cluster_centers, test_preds = cluster(vecs, pca_dim=11, 
                                                       k=13, 
                                                       testvecs=testvecs)
    
    assert isinstance(predictions, np.ndarray)
    assert isinstance(cluster_centers, np.ndarray)
    assert isinstance(test_preds, np.ndarray)
    
    assert predictions.shape == (37,)
    assert predictions.dtype == np.int32
    
    assert test_preds.shape == (5,)
    assert test_preds.dtype == np.int32
    
    assert cluster_centers.shape == (13, 11)
    assert cluster_centers.dtype == np.float64
    
    
def test_build_model():
    prediction_model, training_model, output_layer = _build_model(fcn, 
                                        imshape=(32,32), num_channels=3,
                                        dense=[5,5], k=7)
    
    assert isinstance(prediction_model, tf.keras.Model)
    assert isinstance(training_model, tf.keras.Model)
    assert isinstance(output_layer, tf.keras.layers.Layer)
    
    assert prediction_model.output_shape == (None, 5)
    assert training_model.output_shape == (None, 7)
    
    
    
def test_deepcluster_training_step():
    deepcluster_training_step = build_deepcluster_training_step()
    opt = tf.keras.optimizers.SGD(1e-3)
    x = np.zeros((5,7,11,3))
    y = np.arange(5)
    
    lossdict = deepcluster_training_step(x, y, model, opt)
    assert lossdict["training_crossentropy"].numpy().dtype == np.float32