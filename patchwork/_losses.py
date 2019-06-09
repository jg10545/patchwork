import tensorflow as tf



def entropy_loss(y_true, y_pred):
    """
    Loss function for semi-supervised learning- use this to 
    minimize the entropy of a sigmoid-output network on unlabeled
    data. Pass whatever you want for y_true. Reference:
    
    Semi-supervised Learningby Entropy Minimization by Grandvalet
    and Bengio.
    """
    return tf.keras.losses.categorical_crossentropy(y_pred, y_pred)

