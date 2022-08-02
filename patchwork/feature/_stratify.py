import numpy as np
import tensorflow as tf
import sklearn.cluster

"""
OK, things I need to do every epoch

    -build a prediction model
    -compute projected features for entire dataset
    -compute K-means indices for each training point


"""

def _get_cluster_indices(fcn, ds, k, num_channels=3, proj_dim=128):
    """

    """
    strat = tf.distribute.get_strategy()
    # build prediction model
    with strat.scope():
        inpt = tf.keras.layers.Input((None, None, num_channels))
        net = fcn(inpt)
        net = tf.keras.layers.AvgPool2D()(net)
        if proj_dim is not None:
            net = tf.keras.layers.Dense(proj_dim, use_bias=False)(net)
        net = tf.keras.layers.Normalization()(net)
        model = tf.keras.Model(inpt, net)
    # get features
    features = model.predict(ds)
    # fit a k-means model and return indices
    return sklearn.cluster.KMeans(k).fit_predict(features)





