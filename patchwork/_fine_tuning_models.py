# -*- coding: utf-8 -*-
import param
import tensorflow as tf
import re

from patchwork._layers import _next_layer


def _build_multi_output_fcn(oldfcn, layers, pooling="average pool"):
    """
    Take a Keras feature extractor and
    """
    if pooling == "average pool":
        pool = tf.keras.layers.GlobalAveragePooling2D
    elif pooling == "max pool":
        pool = tf.keras.layers.GlobalMaxPool2D
    else:
        assert False, "what kind of pooling is this?"
    # regex case
    if isinstance(layers, str):
        layers = [e for e, l in enumerate(oldfcn.layers)
                  if len(re.findall(layers, l.name)) > 0]
    outputs = []
    inpt = oldfcn.input
    for l in layers:
        net = oldfcn.layers[l].output
        net = pool()(net)
        outputs.append(net)

    return tf.keras.Model(inpt, outputs)


class GlobalPooling(param.Parameterized):
    """
    Just a single pooling layer.
    """
    pooling_type = param.ObjectSelector(default="max pool", objects=["max pool", "average pool", "flatten"])

    description = """
    A single pooling or flattening layer to map outputs of a feature extractor to a dense vector. No trainable parameters.
    """

    def build(self, feature_shape, feature_extractor):
        #inpt = tf.keras.layers.Input((None, None, inpt_channels))
        inpt = tf.keras.layers.Input(feature_shape)
        if self.pooling_type == "max pool":
            pool = tf.keras.layers.GlobalMaxPool2D()
        elif self.pooling_type == "average pool":
            pool = tf.keras.layers.GlobalAvgPool2D()
        else:
            pool = tf.keras.layers.Flatten()
        # store a reference to the model in case we need it later
        self._model = tf.keras.Model(inpt, pool(inpt), name="finetuning")
        return self._model, feature_extractor

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
    kernel_size = param.ObjectSelector(default=1, objects=[1,3,5,7], doc="Spatial size of filters")
    batchnorm = param.Boolean(False, doc="Whether to use batch normalization in convolutional layers")
    separable_convolutions = param.Boolean(False, doc="Whether to use depthwise separable convolutions")
    dropout_rate = param.Number(0.5, bounds=(0.05,0.95), step=0.05, doc="Spatial dropout rate.")
    pooling_type = param.ObjectSelector(default="max pool", objects=["max pool", "average pool", "flatten"],
                                        doc="Whether to use global mean or max pooling.")


    _description = """
    Convolutional network with global pooling at the end. Set dropout to 0 to disable.
    """
    description = """
    Convolutional network with global pooling at the end. \n\n
    Use a comma-separated list to define layers: integer for a convolution, `p` for 2x2 max pooling, `d` for 2D spatial dropout, and `r` for a residual block.
    """

    def build(self, feature_shape, feature_extractor):
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
        self._model = tf.keras.Model(inpt, net, name="finetuning")
        return self._model, feature_extractor

    def model_params(self):
        return {"fine_tuning_type":"ConvNet",
                "pooling_type":self.pooling_type,
                "num_params":self._model.count_params(),
                "num_layers":len(self._model.layers),
                "kernel_size":self.kernel_size,
                "separable":self.separable_convolutions,
                "structure":self.layers}


class MultiscalePooling(param.Parameterized):
    """
    Concatenate pooled features from multiple layers
    """
    pooling_type = param.ObjectSelector(default="average pool", objects=["max pool", "average pool"])
    layers = param.String(default="add", doc="Comma-separated list of filters")
    use_dropout = param.Boolean(default=True, doc="Use layerwise dropout")
    dropout_rate = param.Number(0.5, bounds=(0.05, 0.95), step=0.05, doc="Spatial dropout rate.")

    description = """
    Loosely based on *Head2Toe: Utilizing Intermediate Representations for Better Transfer Learning* by Evci *et al.*
    Uses layerwise dropout instead of group LASSO. No trainable parameters.
    """

    def build(self, feature_shape, feature_extractor):
        # rebuild the feature extractor to output results at each layer.
        # first parse the layer list
        try:
            layers = [int(x) for x in self.layers.split(",")]
        except:
            layers = self.layers
        # rebuild feature extractor
        multi_output_fcn = _build_multi_output_fcn(feature_extractor,
                                                   layers,
                                                   self.pooling_type)

        # build new finetuning model to accept multiple inputs
        inputshapes = [l.shape[-1] for l in multi_output_fcn.outputs]
        inputs = []
        outlayers = []
        for i in inputshapes:
            inpt = tf.keras.layers.Input(i)
            net = inpt
            if self.use_dropout:
                net = tf.keras.layers.Reshape([i, 1])(net)
                net = tf.keras.layers.SpatialDropout1D(self.dropout_rate)(net)
                net = tf.keras.layers.Reshape([i])(net)
            inputs.append(inpt)
            outlayers.append(net)
        net = tf.keras.layers.Concatenate()(outlayers)
        self._model = tf.keras.Model(inputs, net, name="finetuning")
        return self._model, multi_output_fcn

    def model_params(self):
        return {"fine_tuning_type": "MultiscalePooling",
                "pooling_type": self.pooling_type,
                "num_params": self._model.count_params(),
                "num_layers": len(self._model.layers),
                "feature_layers": self.layers,
                "dropout":self.use_dropout,
                "dropout_rate":self.dropout_rate}

