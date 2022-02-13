import tensorflow as tf

def _residual(inpt, *layers):
    x = inpt
    for l in layers:
        x = l(x)
    return tf.keras.layers.Add()([x, inpt])


def build_convmixer_fcn(dim, depth, kernel_size=9, patch_size=7, gelu=True,
                       input_shape=(None, None, 3)):
    """
    Build a fully-convolutional network based on "Patches are all you need?" by Trockman and Kolter. 
    Follows the implementation in appendix D of the paper, without the final three layers (average pool,
    flatten, linear).
    
    Built without any custom layers/objects so you have the option to save the model with HDF5.
    
    Inputs
    :dim: hidden dimension of each layer of the network
    :depth: number of depthwise/pointwise blocks
    :kernel_size: size of depthwise convolutions
    :patch_size: size of initial patches
    :gelu: if True, use GeLU activations; if False, use ReLU
    :input_shape: shape of initial input tensor

    Returns a keras model
    """
    if gelu:
        activation = tf.nn.gelu
    else:
        activation = tf.nn.relu
    
    inpt = tf.keras.layers.Input(input_shape)
    # patches!
    x = tf.keras.layers.Conv2D(dim, patch_size, strides=patch_size)(inpt)
    x = tf.keras.layers.Activation(activation)(x)
    x = tf.keras.layers.BatchNormalization()(x)

    for d in range(depth):
        # Depthwise convolution with residual
        layers = [tf.keras.layers.DepthwiseConv2D(kernel_size, padding="same"),
                 tf.keras.layers.Activation(activation),
                 tf.keras.layers.BatchNormalization()]
        x = _residual(x, *layers)
        # pointwise convolution
        x = tf.keras.layers.Conv2D(dim, kernel_size=1)(x)
        x = tf.keras.layers.Activation(activation)(x)
        x = tf.keras.layers.BatchNormalization()(x)
    
    return tf.keras.Model(inpt,x)