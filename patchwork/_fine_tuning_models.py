# -*- coding: utf-8 -*-
import param
import tensorflow as tf

from patchwork._layers import _next_layer

class GlobalPooling(param.Parameterized):
    """
    Just a single pooling layer.
    """
    pooling_type = param.ObjectSelector(default="max pool", objects=["max pool", "average pool"])
    
    description = """
    A single pooling layer to map outputs of a feature extractor to a dense vector. No trainable parameters.
    """
    
    def build(self, inpt_channels):
        inpt = tf.keras.layers.Input((None, None, inpt_channels))
        if self.pooling_type == "max pool":
            pool = tf.keras.layers.GlobalMaxPool2D()
        else:
            pool = tf.keras.layers.GlobalAvgPool2D()
        return tf.keras.Model(inpt, pool(inpt))
        
        
class ConvNet(param.Parameterized):
    """
    Convolutional network
    """
    layers = param.String(default="128,p,d,128", doc="Comma-separated list of filters")
    kernel_size = param.ObjectSelector(default=1, objects=[1,3,5], doc="Spatial size of filters")
    separable_convolutions = param.Boolean(False, doc="Whether to use depthwise separable convolutions")
    dropout_rate = param.Number(0.5, bounds=(0.05,0.95), doc="Spatial dropout rate.")
    pooling_type = param.ObjectSelector(default="max pool", objects=["max pool", "average pool"], 
                                        doc="Whether to use global mean or max pooling.")
    
    
    description = """
    Convolutional network with global pooling at the end. Set dropout to 0 to disable.
    """
    
    def build(self, inpt_channels):
        inpt = tf.keras.layers.Input((None, None, inpt_channels))
        net = inpt
        for l in self.layers.split(","):
            l = l.strip()
            net = _next_layer(net, l, kernel_size=self.kernel_size,
                              dropout_rate=self.dropout_rate)
            
        if self.pooling_type == "max pool":
            net = tf.keras.layers.GlobalMaxPool2D()(net)
        else:
            net = tf.keras.layers.GlobalAvgPool2D()(net)
        return tf.keras.Model(inpt, net)