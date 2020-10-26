# -*- coding: utf-8 -*-
"""

            _badge.py

Support code for the BADGE active learning algorithm. See DEEP BATCH ACTIVE 
LEARNING BY DIVERSE, UNCERTAIN GRADIENT LOWER BOUNDS by Ash et al
"""
import numpy as np
import tensorflow as tf
from scipy.spatial.distance import cdist

class KPlusPlusSampler():
    """
    Class for drawing random indices using the initialization
    algorithm from kmeans++
    """
    def __init__(self, X, indices=None):
        """
        :X: (N,d) array of vector
        :indices: initial list of indices (for example, previously-
            labeled records)
        """
        self.X = X
        self.N = X.shape[0]
        self.d = X.shape[1]
        if indices is None:
            indices = []
            
        self.indices = indices
        
        if len(indices) > 0:
            self.min_dists = cdist(X[np.array(indices),:], X).min(axis=0)
        
    def _choose_initial_index(self):
        ind = np.random.randint(self.N)
        self.indices.append(ind)
        self.min_dists = cdist(self.X[np.array([ind]),:], self.X).min(axis=0)
        return ind
    
    def _choose_non_initial_index(self):
        # compute sampling probabilities
        p = self.min_dists**2
        p /= p.sum()
        # sample new index
        ind = np.random.choice(np.arange(self.N), p=p)
        self.indices.append(ind)
        # update min distances
        min_dists = cdist(self.X[np.array([ind]),:], self.X).min(axis=0)
        self.min_dists = np.minimum(self.min_dists, min_dists)
        return ind
    
    def choose(self, k=1):
        """
        Return a list of k sample indices
        """
        indices = []
        for _ in range(k):
            if len(self.indices) == 0:
                ind = self._choose_initial_index()
            else:
                ind = self._choose_non_initial_index()
            indices.append(ind)
            
        return indices
    
    def __call__(self, k=1):
        """
        Return a list of k samples
        """
        return self.choose(k)
        

def _build_output_gradient_function(*models):
    """
    Generate a tensorflow function for computing, for a given example, the gradient of
    the loss function with respect to the weights in the final layer. This is useful
    for active learning- see "DEEP BATCH ACTIVE LEARNING BY DIVERSE, UNCERTAIN GRADIENT 
    LOWER BOUNDS" by Ash et al.
    
    :models: Keras model (or multiple models to be applied sequentially). BADGE gradients
        are computed with respect to kernel weights in the final layer of the last model.
        
    Returns a tensorflow function that maps inputs to flattened BADGE gradients
    """
    # ------------ Identify the weight tensor to compute gradients against -----------
    # find the output layer of the final network
    final_layer = models[-1].layers[-1]
    # in the event that some clown defined this model with nested models,
    # drill down until we get to an actual model
    while isinstance(final_layer, tf.keras.Model):
        final_layer = final_layer.layers[-1]
    
    # THERE SHOULD ONLY BE ONE TENSOR IN THIS LIST
    final_layer_weights = [x for x in final_layer.trainable_variables 
                           if "kernel" in x.name]
    assert len(final_layer_weights) == 1, "not sure which weights to use"
    output_weights = final_layer_weights[0]
    
    # ------------ Define a tf.function -----------
    @tf.function
    def compute_output_gradients(x):
        # ------------ Run input through the model(s) -----------
        pred = x
        with tf.GradientTape() as tape:
            for m in models:
                pred = m(pred)
            # ------------ Create pseudolabel by rounding model predictions -----------
            y = tf.stop_gradient(pred)
            label = tf.cast(y >= 0.5, tf.float32)
            # ------------ Loss between prediction and pseudolabel -----------
            loss = tf.keras.losses.binary_crossentropy(label, pred)
        # ------------ Calculate gradients and return flattened matrix -----------
        grad = tape.jacobian(loss, output_weights)
        return tf.reshape(grad, [x.shape[0], -1])

    return compute_output_gradients

