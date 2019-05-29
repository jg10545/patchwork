"""

                _models.py


param-based code for defining Keras models

"""

import tensorflow as tf
import param
import panel as pn

from patchwork._layers import CosineDense


# base model classes
class LinearSoftmax(param.Parameterized):
    """
    A basic model- input a [batch_size, a, b, num_channels] tensor output from
    a convnet, flatten with a GlobalMaxPool and put through a softmax layer
    """
    
    regularization = param.Parameter(default=0, 
                                doc="L2 norm weight for regularization",
                                label="Regularization Weight")
    
    def _build(self, num_classes, inpt_channels):
        inpt = tf.keras.layers.Input((None, None, inpt_channels))
        net = tf.keras.layers.GlobalMaxPool2D()(inpt)
        if self.regularization > 0:
            reg = tf.keras.regularizers.l2(self.regularization)
        else:
            reg = None
        net = tf.keras.layers.Dense(num_classes, kernel_regularizer=reg,
                                   activation=tf.keras.activations.softmax)(net)
        return tf.keras.Model(inpt, net)


class MLPSoftmax(param.Parameterized):
    """
    The LinearSoftmax model with a hidden layer
    """
    
    regularization = param.Parameter(default=0, 
                                doc="L2 norm weight for regularization",
                                label="Regularization Weight")
    hidden = param.Integer(default=128, bounds=(10,1024), 
                           label="Number of hidden nodes")
    dropout = param.Boolean(default=False, label="Use dropout (0.5)")
    #log_dir = param.Foldername(default="")
    
    def _build(self, num_classes, inpt_channels):
        inpt = tf.keras.layers.Input((None, None, inpt_channels))
        net = tf.keras.layers.GlobalMaxPool2D()(inpt)
        if self.regularization > 0:
            reg = tf.keras.regularizers.l2(self.regularization)
        else:
            reg = None
        net = tf.keras.layers.Dense(self.hidden, kernel_regularizer=reg,
                                   activation=tf.keras.activations.relu)(net)
        if self.dropout:
            net = tf.keras.layers.Dropout(0.5)(net)
        net = tf.keras.layers.Dense(num_classes, kernel_regularizer=reg,
                                   activation=tf.keras.activations.softmax)(net)
        return tf.keras.Model(inpt, net)


class LinearBPP(param.Parameterized):
    """
    The linear model from Chen et al's paper
    """
    
    def _build(self, num_classes, inpt_channels):
        inpt = tf.keras.layers.Input((None, None, inpt_channels))
        net = tf.keras.layers.GlobalMaxPool2D()(inpt)
        net = CosineDense(num_classes)(net)
        return tf.keras.Model(inpt, net)
    
    
model_list = [LinearSoftmax(name="Linear Softmax"), MLPSoftmax(name="Multilayer Perceptron"), 
              LinearBPP(name="Baseline++")]


    