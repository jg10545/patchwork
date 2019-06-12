import tensorflow as tf
from tensorflow.keras import backend as K


def entropy_loss(y_true, y_pred):
    """
    Loss function for semi-supervised learning- use this to 
    minimize the entropy of a sigmoid-output network on unlabeled
    data. Pass whatever you want for y_true. Reference:
    
    Semi-supervised Learning by Entropy Minimization by Grandvalet
    and Bengio.
    """
    return tf.keras.losses.categorical_crossentropy(y_pred, y_pred)




# Based on code at https://www.dlology.com/blog/how-to-multi-task-learning-with-missing-labels-in-keras/

def masked_binary_crossentropy(y_true, y_pred):
    """
    Binary crossentropy wrapper that masks out any values
    where y_true = -1    
    """
    mask = K.cast(K.not_equal(y_true, -1), K.floatx())
    return K.binary_crossentropy(y_true * mask, y_pred * mask)