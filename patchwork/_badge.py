# -*- coding: utf-8 -*-
"""

            _badge.py

Support code for the BADGE active learning algorithm. See DEEP BATCH ACTIVE
LEARNING BY DIVERSE, UNCERTAIN GRADIENT LOWER BOUNDS by Ash et al
"""
import numpy as np
import tensorflow as tf
from sklearn.metrics.pairwise import euclidean_distances

class KPlusPlusSampler():
    """
    Class for drawing random indices using the initialization
    algorithm from kmeans++
    """
    def __init__(self, X, indices=None):
        """


        Parameters
        ----------
        X : (N,d) array
            Matrix containing the point cloud to sample from.
        indices : list, optional
            initial list of previously-sampled indices to
            exclude from sampling. The default is None.

        Returns
        -------
        None.
        """
        self.X = X
        self.N = X.shape[0]
        self.d = X.shape[1]
        # create boolean array of length N indicating which
        # indices are available for sampling
        self.available = np.ones(self.N).astype(bool)
        if indices is not None:
            for i in indices:
                self.available[i] = False
            #indices = []


        #self.indices = [x for x in indices]

        #if len(indices) > 0:
        #    indices = np.array(indices)
        #    self.min_dists_sq = euclidean_distances(X, X[indices], squared=True).min(axis=1)


    def _update_min_dists_sq(self):
        """
        update the array of minimum distances
        """
        # get minimum distance from each unlabeled data point to a labeled data point
        min_dists_sq = euclidean_distances(self.X, self.X[~self.available, :], squared=True).min(axis=1)
        # create or update self.min_dists_sq
        if hasattr(self, "min_dists_sq"):
            self.min_dists_sq = np.minimum(self.min_dists_sq, min_dists_sq)
        else:
            self.min_dists_sq = min_dists_sq

    def _choose_initial_index(self, include=None):
        """
        Pick the first index

        Parameters
        ----------
        include : array, optional
            Boolean array of length N; only sample from indices where
            include[i] == True. The default is None (sample from all indices).

        Returns
        -------
        ind : int
            The sampled index.

        """
        # find available indices
        available = self.available
        if include is not None:
            available = available&include
        indices = np.arange(self.N, dtype=np.int64)[available]
        #if include is not None:
        #    indices = indices[include]
        # randomly pick one
        ind = np.random.choice(indices)
        # record that we picked it
        self.available[ind] = False
        #self.indices.append(ind)
        # compute distances
        self._update_min_dists_sq()
        #self.min_dists_sq = euclidean_distances(self.X, self.X[~self.available,:], squared=True).min(axis=1)
        return int(ind)

    def _choose_non_initial_index(self, include=None):
        """
        Sample any index after the first one

        Parameters
        ----------
        include : array, optional
            Boolean array of length N; only sample from indices where
            include[i] == True. The default is None (sample from all indices).

        Returns
        -------
        ind : int
            The sampled index

        """
        # find available indices
        available = self.available
        if include is not None:
            available = available&include
        indices = np.arange(self.N, dtype=np.int64)[available]
        prob = self.min_dists_sq[available]#**self.p
        # if necessary exclude some
        #if include is not None:
        #    indices = indices[include]
        #    prob = prob[include]
        # make sure we fail gracefully!
        if prob.sum() == 0:
            return None
        # normalize probs
        prob = prob/prob.sum()

        # sample new index
        ind = np.random.choice(indices, p=prob)
        # record that we picked it
        self.available[ind] = False
        #self.indices.append(ind)
        # update min distances
        self._update_min_dists_sq()
        #min_dists_sq = euclidean_distances(self.X, self.X[~self.available,:], squared=True).min(1)
        #self.min_dists_sq = np.minimum(self.min_dists_sq, min_dists_sq)
        return int(ind)

    def choose(self, k=1, include=None):
        """
        Sample k indices

        Parameters
        ----------
        k : int
            Number of indices to sample
        include : array, optional
            Boolean array of length N; only sample from indices where
            include[i] == True. The default is None (sample from all indices).

        Returns
        -------
        indices : list of ints
            The sampled indices.

        """
        indices = []
        for _ in range(k):
            if hasattr(self, "min_dists_sq"):
                ind = self._choose_non_initial_index(include)
            else:
                ind = self._choose_initial_index(include)
            #if len(self.indices) == 0:
            #    ind = self._choose_initial_index(include)
            #else:
            #    ind = self._choose_non_initial_index(include)
            if ind is not None:
                indices.append(ind)

        return indices

    def __call__(self, k=1, include=None):
        """
        Return a list of k samples
        """
        return self.choose(k, include)


def _build_output_gradient_function(*models, select_output=None):
    """
    Generate a tensorflow function for computing, for a given example, the gradient of
    the loss function with respect to the weights in the final layer. This is useful
    for active learning- see "DEEP BATCH ACTIVE LEARNING BY DIVERSE, UNCERTAIN GRADIENT
    LOWER BOUNDS" by Ash et al.

    :models: Keras model (or multiple models to be applied sequentially). BADGE gradients
        are computed with respect to kernel weights in the final layer of the last model.
    :select_output: if final model has a multiclass output, select which class to compute
        gradients from

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

        if select_output is not None:
            grad = grad[:,:,select_output]
        return tf.reshape(grad, [x.shape[0], -1])

    return compute_output_gradients

