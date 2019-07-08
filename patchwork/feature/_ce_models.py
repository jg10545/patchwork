"""

            _ce_models.py

Code to generate model pieces to use with a context encoder.
"""
import tensorflow as tf

def build_encoder(num_channels=3):
    """
    Inpainting encoder model from Pathak et al
    """
    inpt = tf.keras.layers.Input((None, None, num_channels))
    net = inpt
    for k in [64, 64, 128, 256, 512]:
        net = tf.keras.layers.Conv2D(k, 4, strides=2, padding="same")(net)
        net = tf.keras.layers.LeakyReLU(alpha=0.2)(net)
        net = tf.keras.layers.BatchNormalization()(net)
    return tf.keras.Model(inpt, net, name="encoder")


def build_decoder(input_channels=512, num_channels=3):
    """
    Inpainting decoder from Pathak et al
    """
    inpt = tf.keras.layers.Input((None, None, input_channels))
    net = inpt

    for k in [512, 256, 128, 64, 64]:
        net = tf.keras.layers.Conv2DTranspose(k, 4, strides=2, 
                                              padding="same",
                                activation=tf.keras.activations.relu)(net)
        net = tf.keras.layers.BatchNormalization()(net)
    
    net = tf.keras.layers.Conv2D(num_channels, 3, strides=1, padding="same", 
                             activation=tf.keras.activations.sigmoid)(net)
    return tf.keras.Model(inpt, net, name="decoder")


def build_discriminator(num_channels=3):
    """
    Inpainting discriminator from Pathak et al
    """
    inpt = tf.keras.layers.Input((None, None, num_channels))
    net = inpt
    for k in [64, 128, 256, 512]:
        net = tf.keras.layers.Conv2D(k, 4, strides=2, padding="same",
                                activation=tf.keras.activations.relu)(net)
        net = tf.keras.layers.BatchNormalization()(net)
    net = tf.keras.layers.GlobalMaxPool2D()(net)
    net = tf.keras.layers.Dense(1, activation=tf.keras.activations.sigmoid,
                               name="disc_pred")(net)
    return tf.keras.Model(inpt, net, name="discriminator")
