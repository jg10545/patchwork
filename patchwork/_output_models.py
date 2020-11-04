# -*- coding: utf-8 -*-
import param
import tensorflow as tf

from patchwork._losses import masked_binary_crossentropy
from patchwork._losses import masked_binary_focal_loss
#from patchwork._losses import masked_mean_average_error

from patchwork._layers import CosineDense

def _norm(x):
    import tensorflow as tf
    return tf.keras.backend.l2_normalize(x,1)

class SigmoidCrossEntropy(param.Parameterized):
    """
    Output network for the basic sigmoid case
    """
    normalize = param.Boolean(default=False, doc="whether to L2-normalize inputs")
    label_smoothing = param.Number(0, bounds=(0, 0.25), step=0.05, 
                                   doc="epsilon for label smoothing")
    
    
    description = """
    Use a sigmoid function to estimate class probabilities and use cross-entropy loss to train.
    """
    
    def build(self, num_classes, inpt_channels):
        # return output model as well as loss function
        inpt = tf.keras.layers.Input((inpt_channels))
        net = inpt
        if self.normalize:
            net = tf.keras.layers.Lambda(_norm)(net)
        dense = tf.keras.layers.Dense(num_classes, activation="sigmoid")(net)
        def loss(y_true, y_pred):
            return masked_binary_crossentropy(y_true, y_pred, label_smoothing=self.label_smoothing)
        self._model = tf.keras.Model(inpt, dense)
        return self._model, loss
    
    def model_params(self):
        return {"output_type":"CrossEnt",
                "normalize":self.normalize,
                "label_smoothing":self.label_smoothing,
                "num_params":self._model.count_params()}
    
class SigmoidFocalLoss(param.Parameterized):
    """
    Output network for the basic sigmoid case
    """
    normalize = param.Boolean(default=False, doc="whether to L2-normalize inputs")
    gamma = param.Number(2., doc="focal loss gamma parameter")
    
    description = """
    Use a sigmoid function to estimate class probabilities and use focal loss to train, putting more emphasis on difficult cases. See "Focal Loss for Dense Object Detection" by Lin et al.
    """
    
    def build(self, num_classes, inpt_channels):
        # return output model as well as loss function
        inpt = tf.keras.layers.Input((inpt_channels))
        net = inpt
        if self.normalize:
            net = tf.keras.layers.Lambda(_norm)(net)
        dense = tf.keras.layers.Dense(num_classes, activation="sigmoid")(net)
        def loss(y_true, y_pred):
            return masked_binary_focal_loss(y_true, y_pred, self.gamma)
        self._model = tf.keras.Model(inpt, dense)
        return self._model, loss
    
    def model_params(self):
        return {"output_type":"FocalLoss",
                "normalize":self.normalize,
                "gamma":self.gamma,
                "num_params":self._model.count_params()}
    

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
        
        self._model = tf.keras.Model(inpt, dense)
        return self._model, masked_binary_crossentropy
    
    def model_params(self):
        return {"output_type":"Cosine",
                "num_params":self._model.count_params()}