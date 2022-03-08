"""

            _models.py

Model definitions to use for building feature extractors
"""
import tensorflow as tf

_alex_layers = [
        (96, 11, 4),
        "M",
        (256, 5, 1),
        "M",
        (384, 3, 1),
        (384, 3, 1),
        (256, 3, 1),
        "M"
        ]

def BNAlexNetFCN(num_channels=3):
    """
    Like the conv layers of AlexNet, using Batch Norm instead of LRN.
    
    :num_channels: number of channels for input image.
    """
    inpt = tf.keras.layers.Input((None, None, num_channels))
    net = inpt
    for l in _alex_layers:
        if l == "M":
            net = tf.keras.layers.MaxPool2D(3,2)(net)
        else:
            k, w, s = l
            # changed same to valid
            net = tf.keras.layers.Conv2D(k, w, strides=s, padding="same",
                                     activation="relu")(net)
            net = tf.keras.layers.BatchNormalization()(net)
    return tf.keras.Model(inpt, net)
    


    

def build_encoder(num_channels=3):
    """
    Inpainting encoder model from Pathak et al
    """
    inpt = tf.keras.layers.Input((None, None, num_channels))
    net = inpt
    #for k in [64, 64, 128, 256, 512]:
    for k in [32, 64, 128, 256, 512]:
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

    #for k in [512, 256, 128, 64, 64]:
    for k in [512, 256, 128, 64, 32]:
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