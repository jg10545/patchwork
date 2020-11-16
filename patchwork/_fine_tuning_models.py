# -*- coding: utf-8 -*-
import param
import tensorflow as tf

from patchwork._layers import _next_layer


class GlobalPooling(param.Parameterized):
    """
    Just a single pooling layer.
    """
    pooling_type = param.ObjectSelector(default="max pool", objects=["max pool", "average pool", "flatten"])
    
    description = """
    A single pooling or flattening layer to map outputs of a feature extractor to a dense vector. No trainable parameters.
    """
    
    def build(self, feature_shape):
        #inpt = tf.keras.layers.Input((None, None, inpt_channels))
        inpt = tf.keras.layers.Input(feature_shape)
        if self.pooling_type == "max pool":
            pool = tf.keras.layers.GlobalMaxPool2D()
        elif self.pooling_type == "average pool":
            pool = tf.keras.layers.GlobalAvgPool2D()
        else:
            pool = tf.keras.layers.Flatten()
        # store a reference to the model in case we need it later
        self._model = tf.keras.Model(inpt, pool(inpt))
        return self._model
    
    def model_params(self):
        return {"fine_tuning_type":"GlobalPooling",
                "pooling_type":self.pooling_type,
                "num_params":self._model.count_params(),
                "num_layers":len(self._model.layers)}
        
        
class ConvNet(param.Parameterized):
    """
    Convolutional network
    """
    layers = param.String(default="128,p,d,128", doc="Comma-separated list of filters")
    kernel_size = param.ObjectSelector(default=1, objects=[1,3,5], doc="Spatial size of filters")
    batchnorm = param.Boolean(False, doc="Whether to use batch normalization in convolutional layers")
    separable_convolutions = param.Boolean(False, doc="Whether to use depthwise separable convolutions")
    dropout_rate = param.Number(0.5, bounds=(0.05,0.95), doc="Spatial dropout rate.")
    pooling_type = param.ObjectSelector(default="max pool", objects=["max pool", "average pool", "flatten"], 
                                        doc="Whether to use global mean or max pooling.")
    
    
    description = """
    Convolutional network with global pooling at the end. Set dropout to 0 to disable.
    """
    
    def build(self, feature_shape):
        inpt = tf.keras.layers.Input(feature_shape)
        net = inpt
        for l in self.layers.split(","):
            l = l.strip()
            net = _next_layer(net, l, kernel_size=self.kernel_size,
                              dropout_rate=self.dropout_rate,
                              separable=self.separable_convolutions,
                              batchnorm=self.batchnorm)
            
        if self.pooling_type == "max pool":
            net = tf.keras.layers.GlobalMaxPool2D()(net)
        elif self.pooling_type == "average pool":
            net = tf.keras.layers.GlobalAvgPool2D()(net)
        else:
            net = tf.keras.layers.Flatten()(net)
        # store reference to model in case we need it later
        self._model = tf.keras.Model(inpt, net)
        return self._model
    
    def model_params(self):
        return {"fine_tuning_type":"ConvNet",
                "pooling_type":self.pooling_type,
                "num_params":self._model.count_params(),
                "num_layers":len(self._model.layers),
                "kernel_size":self.kernel_size,
                "separable":self.separable_convolutions,
                "structure":self.layers}
        