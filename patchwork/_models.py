"""

                _models.py


param-based code for defining Keras models

"""

import tensorflow as tf
import param
import panel as pn

from patchwork._layers import CosineDense


# base model classes
class Linear(param.Parameterized):
    """
    A basic model- input a [batch_size, a, b, num_channels] tensor output from
    a convnet, flatten with a GlobalMaxPool and put through a softmax or sigmoid layer
    """
    
    regularization = param.Parameter(default=0, 
                                doc="L2 norm weight for regularization",
                                label="Regularization Weight")
    
    description = """
    A simple linear model: apply a 2D global max pool to the input
    tensor, then pass through a single output layer.
    """
    
    def _build(self, num_classes, inpt_channels):
        inpt = tf.keras.layers.Input((None, None, inpt_channels))
        net = tf.keras.layers.GlobalMaxPool2D()(inpt)
        if self.regularization > 0:
            reg = tf.keras.regularizers.l2(self.regularization)
        else:
            reg = None
        net = tf.keras.layers.Dense(num_classes, kernel_regularizer=reg,
                                   activation=tf.keras.activations.sigmoid)(net)
        return tf.keras.Model(inpt, net)


class MLP(param.Parameterized):
    """
    The Linear model with a hidden layer
    """
    
    regularization = param.Parameter(default=0, 
                                doc="L2 norm weight for regularization",
                                label="Regularization Weight")
    hidden = param.Integer(default=128, bounds=(10,1024), 
                           label="Number of hidden nodes")
    dropout = param.Boolean(default=False, label="Use dropout (0.5)")
    #log_dir = param.Foldername(default="")
    description = """
    Apply a 2D global max pool to the input tensor, then a hidden layer
    (with ReLU activation) before the output layer.
    """
    
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
                                   activation=tf.keras.activations.sigmoid)(net)
        return tf.keras.Model(inpt, net)


class LinearBPP(param.Parameterized):
    """
    The linear model from Chen et al's paper
    """
    description = """
    The modified linear model from Chen et al's paper
    """
    
    def _build(self, num_classes, inpt_channels):
        inpt = tf.keras.layers.Input((None, None, inpt_channels))
        net = tf.keras.layers.GlobalMaxPool2D()(inpt)
        net = CosineDense(num_classes, activation=tf.keras.activations.sigmoid)(net)
        return tf.keras.Model(inpt, net)
    
    
model_list = [Linear(name="Linear Softmax"), MLP(name="Multilayer Perceptron"), 
              LinearBPP(name="Baseline++")]


model_dict = {"Linear":Linear(name="Linear"), 
              "Multilayer Perceptron":MLP(name="Multilayer Perceptron"), 
              "Baseline++":LinearBPP(name="Baseline++")}



    