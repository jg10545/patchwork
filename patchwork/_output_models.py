# -*- coding: utf-8 -*-
import param
import tensorflow as tf

from patchwork._losses import masked_binary_crossentropy
from patchwork._losses import masked_mean_average_error

from patchwork._layers import CosineDense

class SigmoidCrossEntropy(param.Parameterized):
    """
    Output network for the basic sigmoid case
    """
    label_smoothing = param.Number(0, bounds=(0, 0.5), doc="epsilon for label smoothing")
    
    
    description = """
    Use a sigmoid function to estimate class probabilities and use cross-entropy loss to train.
    """
    
    def build(self, num_classes, inpt_channels):
        # return output model as well as loss function
        inpt = tf.keras.layers.Input((None, inpt_channels))
        dense = tf.keras.layers.Dense(num_classes, activation="sigmoid")(inpt)
        def loss(y_true, y_pred):
            return masked_binary_crossentropy(y_true, y_pred, label_smoothing=self.label_smoothing)
        return tf.keras.Model(inpt, dense), loss
    

class CosineOutput(param.Parameterized):
    """
    Output network that estimates class probabilities using cosine similarity.
    """
    
    description = """
    Use cosine similarity to a set of class embeddings to estimate class probabilities.
    """
    
    def build(self, num_classes, inpt_channels):
        # return output model as well as loss function
        inpt = tf.keras.layers.Input((inpt_channels))
        dense = CosineDense(num_classes)(inpt)
        
        return tf.keras.Model(inpt, dense), masked_mean_average_error