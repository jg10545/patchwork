# -*- coding: utf-8 -*-
import tensorflow as tf


_channels = {
    "T":(96, 192, 384, 768),
    "S": (96, 192, 384, 768),
    "B":(128, 256, 512, 1024),
    "L":(192, 384, 768, 1536),
    "XL":(256, 512, 1024, 2048)
}

_blocks = {
    "T":(3, 3, 9, 3),
    "S":(3, 3, 27, 3),
    "B":(3, 3, 27, 3),
    "L":(3, 3, 27, 3),
    "XL":(3, 3, 27, 3)
}


def _add_convnext_block(inpt, **kwargs):
    """
    
    """
    k0 = inpt.shape[-1]
    x = tf.keras.layers.DepthwiseConv2D(7, padding="same")(inpt)
    x = tf.keras.layers.LayerNormalization()(x)
    
    x = tf.keras.layers.Conv2D(4*k0, 1)(x)
    x = tf.keras.layers.Activation(tf.nn.gelu)(x)
    
    x = tf.keras.layers.Conv2D(k0,1)(x)
    return tf.keras.layers.Add(**kwargs)([inpt,x])


def build_convnext_fcn(m, num_channels=3):
    """
    :m: str; "T", "S", "B", "L", or "XL"
    """
    inpt = tf.keras.layers.Input((None, None, num_channels))
    x = tf.keras.layers.Conv2D(_channels[m][0], 4, strides=4)(inpt)
    x = tf.keras.layers.LayerNormalization()(x)
    
    for stage in range(4):
        if stage == 3:
            kwargs = {"dtype":"float32"}
        else:
            kwargs = {}
        # add stage of convnext blocks
        for i in range(_blocks[m][stage]):
            x = _add_convnext_block(x, **kwargs)
        if stage < 3:
            # add downsampling layer
            x = tf.keras.layers.LayerNormalization()(x)
            x = tf.keras.layers.Conv2D(_channels[m][stage+1], 2, strides=2)(x)
        
    return tf.keras.Model(inpt, x)
