# -*- coding: utf-8 -*-
import param
import tensorflow as tf

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
            # MAX POOL LAYER
            if l.lower() == "p":
                net = tf.keras.layers.MaxPool2D(2,2)(net)
            # DROPOUT   LAYER
            elif l.lower() == "d":   
                net = tf.keras.layers.SpatialDropout2D(self.dropout_rate)(net)
            # CONVOLUTION LAYER
            else:
                num_filters = int(l)
                # Choose whether to add a dropout layer first,
                # as well as whether to use normal or separable
                # convolutions
                #if self.dropout_rate > 0:
                #     net = tf.keras.layers.SpatialDropout2D(self.dropout_rate)(net)
                if self.separable_convolutions:
                    net = tf.keras.layers.SeparableConvolution2D(num_filters,
                                                            self.kernel_size,
                                                            padding="same",
                                                            activation="relu")(net)
                else:
                    net = tf.keras.layers.Conv2D(num_filters,
                                             self.kernel_size,
                                             padding="same",
                                             activation="relu")(net)
                
        if self.pooling_type == "max pool":
            net = tf.keras.layers.GlobalMaxPool2D()(net)
        else:
            net = tf.keras.layers.GlobalAvgPool2D()(net)
        return tf.keras.Model(inpt, net)