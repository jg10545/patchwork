# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf

from patchwork.feature._simsiam import _simsiam_loss, _build_embedding_models

inpt = tf.keras.layers.Input((None, None, 3))
net = tf.keras.layers.Conv2D(5, 1)(inpt)
fcn = tf.keras.Model(inpt, net)

def test_simsiam_loss():
    N = 3
    d = 5
    
    p = np.random.normal(0, 1, (N,d))
    z = np.random.normal(0, 1, (N,d))
    
    loss1 = _simsiam_loss(p, z)
    assert loss1.shape == ()
    
    loss2 = _simsiam_loss(p,p)
    assert round(loss2.numpy(),5) == -1


def test_build_embedding_models():
    num_hidden = 23
    pred_dim = 17
    
    project, predict = _build_embedding_models(fcn, (13,13), 3,
                                               num_hidden=num_hidden, 
                                               pred_dim=pred_dim)
    
    inpt = np.zeros((1,13,13,3), dtype=np.float32)
    project_out = project(inpt)
    predict_out = predict(project_out)
    
    assert isinstance(project, tf.keras.Model)
    assert isinstance(predict, tf.keras.Model)
    assert project_out.shape == (1, 23)
    assert predict_out.shape == (1,23)
    
    
    