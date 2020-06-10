# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf

from patchwork._losses import entropy_loss, masked_binary_crossentropy
from patchwork._losses import masked_mean_average_error
from patchwork._losses import masked_binary_focal_loss

def test_entropy_loss():
    bs = 7
    nc = 5
    pred1 = 0.5*np.ones((bs, nc))
    pred2 = np.zeros((bs,nc))
    
    ent1 = entropy_loss(pred1)
    ent2 = entropy_loss(pred2)
    
    assert isinstance(ent1, tf.Tensor)
    assert np.round(ent1.numpy().mean(),5) == np.round(np.log(2),5)
    assert ent2.numpy().sum() == 0
    
    
def test_masked_binary_crossentropy():
    y_true = np.array([-1., -1, 0., 1.])
    y_pred = np.array([1., 0., 0., 1.])
    
    mbc = masked_binary_crossentropy(y_true, y_pred)
        
    assert isinstance(mbc, tf.Tensor)
    assert mbc.numpy() < 1e-5
    

def test_masked_focal_loss():
    y_true = np.array([-1., -1, 0., 1.], dtype=np.float32)
    y_pred = np.array([1., 0., 0.25, 0.75], dtype=np.float32)
    
    mbc = masked_binary_crossentropy(y_true, y_pred)
    focal_loss_gamma_0 = masked_binary_focal_loss(y_true, y_pred, 0)
    focal_loss_gamma_2 = masked_binary_focal_loss(y_true, y_pred, 2)
        
    assert isinstance(focal_loss_gamma_0, tf.Tensor)
    # at gamma=0, focal loss collapses to crossentropy loss
    assert abs(mbc.numpy()-focal_loss_gamma_0.numpy()) < 1e-5
    # at higher gamma values the loss is pushed down
    assert mbc.numpy() > focal_loss_gamma_2.numpy()
    
    
def test_masked_mean_average_error():
    y_true = np.array([[0,1,-1],[-1,0,1]])
    y_pred = np.array([[0,1,0], [1,1,0]])
    
    mae = masked_mean_average_error(y_true, y_pred).numpy()
    mae2 = masked_mean_average_error(y_true, y_true).numpy()
    
    #assert (mae == np.array([0, 0.5])).all()
    #assert (mae2 == np.array([0, 0])).all()
    assert mae == 0.5
    assert mae2 == 0.