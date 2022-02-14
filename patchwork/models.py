# -*- coding: utf-8 -*-
import tensorflow as tf
from patchwork._convmixer import build_convmixer_fcn

def build_wide_resnet(n=16, k=1, num_channels=3, dropout=0.5, inputshape=None,
                      syncbn=False):
    """
    Build a WideResnet WRN-n-k, as defined in "Wide Residual Networks" by Zagoruyko et al.
    
    Does not include the final average pooling layer.
    
    :n: int; number of conv layers in network (10, 16, 2, 28, 34...)
    :k: int; width scaling of network
    :num_channels: int; number of input channels for fully-convolutional network
    :dropout: float; dropout probability (see figure 1(d) in the paper). 0 to disable.
    :inputshape: 3-tuple; shape of input layer for fixed-size network (e.g. (128,128,3))
    :syncbn: whether to use SynchronizedBN
    """
    assert (n-4)%6 == 0, "invalid number of conv layers (try 10, 16, 22, 28, 34...)"
    N = int((n-4)/6)
    
    if syncbn:
        BN = tf.keras.layers.experimental.SyncBatchNormalization
    else:
        BN = tf.keras.layers.BatchNormalization

    #inpt = tf.keras.layers.Input((None, None, num_channels))
    if inputshape is None:
        inputshape = (None, None, num_channels)
    inpt = tf.keras.layers.Input(inputshape)
    net = inpt

    # conv1
    net = tf.keras.layers.Conv2D(16, 3, padding="same")(net)
    net = BN()(net)
    net = tf.keras.layers.Activation("relu")(net)

    # conv2, conv3, and conv4
    for b in [16, 32, 64]:
        # N residual blocks per group
        for i in range(N):
            block_start = net
            # batchnorm and relu before convolution
            net = BN()(net)
            net = tf.keras.layers.Activation("relu")(net)
            # spatial downsample on the first conv of every group
            if i == 0:
                block_start = tf.keras.layers.Conv2D(b*k, 3, padding="same", 
                                                 strides=2, activation="relu")(block_start)
                net = tf.keras.layers.Conv2D(b*k, 3, padding="same", strides=2)(net)
            else:
                net = tf.keras.layers.Conv2D(b*k, 3, padding="same")(net)
            
            if dropout > 0:
                net = tf.keras.layers.SpatialDropout2D(dropout)(net)
            
            net = BN()(net)
            net = tf.keras.layers.Activation("relu")(net)
            net = tf.keras.layers.Conv2D(b*k, 3, padding="same")(net)
            # end residual block
            net = tf.keras.layers.Add()([block_start, net])
            net = BN()(net)
        
    return tf.keras.Model(inpt, net)
