# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf

from patchwork._badge import KPlusPlusSampler, _build_output_gradient_function

def test_KPlusPlusSampler():
    N = 100
    data = np.random.uniform(0,1, size=(N,2))
    
    sampler = KPlusPlusSampler(data)
    
    # SAMPLE ONE INDEX
    ind = sampler()
    assert isinstance(ind, list)
    assert len(ind) == 1
    assert isinstance(ind[0], int)
    
    # SAMPLE SEVERAL INDICES
    inds = sampler(5)
    assert isinstance(inds, list)
    assert len(inds) == 5
    assert isinstance(ind[0], int)
    assert len(sampler.indices) == 6
    
    

def test_KPlusPlusSampler_with_include_array():
    N = 100
    data = np.random.uniform(0,1, size=(N,2))
    include = np.random.choice([False, True], size=N)
    
    sampler = KPlusPlusSampler(data)
    
    # SAMPLE ONE INDEX
    ind = sampler(include=include)
    assert isinstance(ind, list)
    assert len(ind) == 1
    assert isinstance(ind[0], int)
    
    # SAMPLE SEVERAL INDICES
    inds = sampler(5, include)
    assert isinstance(inds, list)
    assert len(inds) == 5
    assert isinstance(ind[0], int)
    assert len(sampler.indices) == 6
    
def test_KPlusPlusSampler_fails_politely():
    N = 10
    data = np.random.uniform(0, 1, size=(N,2))
    include = np.array([False]*7 + [True]*3)
    
    sampler = KPlusPlusSampler(data)
    # sample more indices than we have available
    inds = sampler(4, include)
    assert isinstance(inds, list)
    assert len(inds) == 3
    
def test_build_output_gradient_function():
    # simple mock fine-tuning network
    inpt = tf.keras.layers.Input((None, None, 8))
    net = tf.keras.layers.GlobalMaxPooling2D()(inpt)
    ft_model = tf.keras.Model(inpt, net)
    # simple mock output network, 3 classes
    inpt = tf.keras.layers.Input((8,))
    net = tf.keras.layers.Dense(3, activation="sigmoid")(inpt)
    o_model = tf.keras.Model(inpt, net)
    
    grad_func = _build_output_gradient_function(ft_model, o_model)
    
    # data shaped like feature tensors. do a batch of 7
    data = tf.random.normal((7,2,2,8), 0, 1, dtype=tf.float32)
    
    # run data through the function
    output_gradients = grad_func(data)
    
    assert isinstance(output_gradients, tf.Tensor)
    assert output_gradients.shape == (7,3*8)

