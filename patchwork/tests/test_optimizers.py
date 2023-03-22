# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
tf.random.set_seed(1)
from patchwork._optimizers import CosineDecayWarmup, LARSOptimizer


def test_cosinedecaywarmup():
    lr = 0.01
    decay_steps = 100000
    warmup_steps = 1000
    alpha = 0.01
    schedule = CosineDecayWarmup(lr, decay_steps,
                                 warmup_steps=warmup_steps,
                                 alpha=alpha)
    # start at 0
    assert schedule(0) == 0
    # linear warmup
    assert schedule(warmup_steps/2) == lr/2
    assert schedule(warmup_steps) == lr
    # decay to alpha*lr
    assert schedule(10*decay_steps) == lr*alpha


def test_lars_optimizer():
    inpt = tf.keras.layers.Input((5))
    net = tf.keras.layers.Dense(1, activation="sigmoid")(inpt)
    model = tf.keras.Model(inpt, net)
    x = tf.zeros(shape=(1,5), dtype=tf.float32)
    y = tf.zeros(shape=(1,1), dtype=tf.float32)
    
    opt = LARSOptimizer(0.01)
    
    with tf.GradientTape() as tape:
        pred = model(x)
        loss = tf.reduce_mean((pred-y)**2)
        
    gradients = tape.gradient(loss, model.trainable_variables)
    opt.apply_gradients(zip(gradients,
                                      model.trainable_variables))
    assert "LARS" in opt._name
    
    
    