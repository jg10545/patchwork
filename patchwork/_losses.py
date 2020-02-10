import tensorflow as tf
from tensorflow.keras import backend as K


def entropy_loss(y_pred):
    """
    Loss function for semi-supervised learning- use this to 
    minimize the entropy of a sigmoid-output network on unlabeled
    data. Pass whatever you want for y_true. Reference:
    
    Semi-supervised Learning by Entropy Minimization by Grandvalet
    and Bengio.
    """
    #return tf.keras.losses.categorical_crossentropy(y_pred, y_pred)
    # for our multi-hot encoding:
    return K.mean(tf.keras.losses.binary_crossentropy(y_pred, y_pred))


# Based on code at https://www.dlology.com/blog/how-to-multi-task-learning-with-missing-labels-in-keras/

def masked_binary_crossentropy(y_true, y_pred, label_smoothing=0):
    """
    Binary crossentropy wrapper that masks out any values
    where y_true = -1    
    """
    mask = K.cast(K.not_equal(y_true, -1), K.floatx())
    # count number of nonempty masks so that we can
    # compute the mean
    norm = K.sum(mask)

    y_true = K.cast(y_true, K.floatx())
    if label_smoothing > 0:
        y_true = y_true*(1-label_smoothing) + 0.5*label_smoothing
    return K.sum(
            K.binary_crossentropy(y_true * mask, y_pred * mask))/norm
    
    
def masked_mean_average_error(y_true, y_pred):
    """
    Mean average error loss function that masks 
    out any values where y_true = -1    
    """ 
    # mask ==1 wherever the label != -1
    mask = K.cast(K.not_equal(y_true, -1), K.floatx())
    y_true = K.cast(y_true, K.floatx())
    y_pred = K.cast(y_pred, K.floatx())
    # count number of nonempty masks so that we can
    # compute the mean
    norm = K.sum(mask)
    return K.sum(
            K.abs(y_true*mask - y_pred*mask))/norm