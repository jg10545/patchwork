"""

            _models.py

Model definitions to use for building feature extractors
"""
import tensorflow as tf


_alex_layers = [
        (96, 11, 4),
        (256, 5, 1),
        (384, 3, 1),
        (384, 3, 1),
        (256, 3, 1)
        ]

def LeakyBNAlexNetFCN(num_channels=3):
    """
    Like the conv layers of AlexNet, but with strided convolutions
    instead of max pooling, batch norms, and LeakyReLU activation.
    
    :num_channels: number of channels for input image.
    """
    inpt = tf.keras.layers.Input((None, None, num_channels))
    net = inpt
    for k, w, s in _alex_layers:
        net = tf.keras.layers.Conv2D(k, w, strides=s, padding="same")(net)
        net = tf.keras.layers.LeakyReLU(alpha=0.2)(net)
        net = tf.keras.layers.BatchNormalization()(net)
        
    return tf.keras.Model(inpt, net)
    