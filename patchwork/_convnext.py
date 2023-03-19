# -*- coding: utf-8 -*-
import tensorflow as tf


_channels = {
    "A":(40, 80, 160, 320), #v2 atto
    "F":(48, 96, 192, 384), #v2 femto
    "P":(64, 128, 256, 512), #v2 pico
    "N":(80, 160, 320, 640), #v2 nano
    "T":(96, 192, 384, 768),
    "S": (96, 192, 384, 768),
    "B":(128, 256, 512, 1024),
    "L":(192, 384, 768, 1536),
    "XL":(256, 512, 1024, 2048)
}

_blocks = {
    "A":(2, 2, 6, 2),
    "F":(2, 2, 6, 2),
    "P":(2, 2, 6, 2),
    "N":(2, 2, 8, 2),
    "T":(3, 3, 9, 3),
    "S":(3, 3, 27, 3),
    "B":(3, 3, 27, 3),
    "L":(3, 3, 27, 3),
    "XL":(3, 3, 27, 3)
}

class GRN(tf.keras.layers.Layer):
    """
    Global Response Normalization layer
    ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders (https://arxiv.org/abs/2301.00808)
    https://github.com/facebookresearch/ConvNeXt-V2/blob/2553895753323c6fe0b2bf390683f5ea358a42b9/models/utils.py#L105

    pytorch original:
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(1,2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x

    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        n_channels = input_shape[-1]
        self.gamma = self.add_weight(
            shape=(n_channels,),
            initializer="zeros",
            name="gamma",
            trainable=True,
        )
        self.beta = self.add_weight(
            shape=(n_channels,),
            initializer="zeros",
            name="beta",
            trainable=True,
        )

    def call(self, inputs):
        Gx = tf.norm(inputs, ord=2, axis=(1,2), keepdims=True)
        Nx = Gx / (tf.reduce_mean(Gx, axis=-1, keepdims=True) + 1e-6) # CN2 code uses mean, paper seems to use sum
        return self.gamma * (inputs * Nx) + self.beta + inputs

    def get_config(self):
        config = super().get_config()
        return config


def _add_convnext_block(inpt, use_grn, **kwargs):
    """
    
    """
    k0 = inpt.shape[-1]
    x = tf.keras.layers.DepthwiseConv2D(7, padding="same")(inpt)
    x = tf.keras.layers.LayerNormalization()(x)
    
    x = tf.keras.layers.Conv2D(4*k0, 1)(x)
    x = tf.keras.layers.Activation(tf.nn.gelu)(x)

    if use_grn:
        x = GRN()(x)
    
    x = tf.keras.layers.Conv2D(k0,1)(x)
    return tf.keras.layers.Add(**kwargs)([inpt,x])


def build_convnext_fcn(m, use_grn=False, num_channels=3):
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
            x = _add_convnext_block(x, use_grn, **kwargs)
        if stage < 3:
            # add downsampling layer
            x = tf.keras.layers.LayerNormalization()(x)
            x = tf.keras.layers.Conv2D(_channels[m][stage+1], 2, strides=2)(x)
        
    return tf.keras.Model(inpt, x)
