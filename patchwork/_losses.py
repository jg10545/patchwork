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
    norm = K.sum(mask) + K.epsilon()

    y_true = K.cast(y_true, K.floatx())
    if label_smoothing > 0:
        y_true = y_true*(1-label_smoothing) + 0.5*label_smoothing
    return K.sum(
            K.binary_crossentropy(y_true * mask, y_pred * mask))/norm


def masked_binary_focal_loss(y_true, y_pred, gamma=2.):
    """
    Focal loss, from "Focal Loss for Dense Object Detection"
    by Lin et al. Currently only implementing the gamma
    parameter, not alpha.
    
    Masks out any values where y_true = -1
    """
    mask = K.cast(K.not_equal(y_true, -1), K.floatx())
    # count number of nonempty masks so that we can
    # compute the mean
    norm = K.sum(mask) + K.epsilon()
    
    loss = 0
    
    # positive cases
    pos = K.cast(K.equal(y_true, 1), K.floatx())
    loss -= K.sum(pos*K.log(y_pred + K.epsilon())*(1-y_pred)**gamma)
    # negative cases
    neg = K.cast(K.equal(y_true, 0), K.floatx())
    loss -= K.sum(neg*K.log(1-y_pred + K.epsilon())*(y_pred)**gamma)
    
    return loss/norm
    
    
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
    norm = K.sum(mask) + K.epsilon()
    return K.sum(
            K.abs(y_true*mask - y_pred*mask))/norm
            
            
def masked_sparse_categorical_crossentropy(y_true, y_pred):
    """
    Sparse categorical crossentropy wrapper that masks out any values
    where y_true = -1    
    """
    mask = K.cast(K.not_equal(y_true, -1), K.floatx())
    # count number of nonempty masks so that we can
    # compute the mean
    norm = K.sum(mask) + K.epsilon()

    y_true_clipped = K.cast(tf.clip_by_value(y_true, 0, 10000), tf.int64)
    #y_true = K.cast(y_true, K.floatx())
    #if label_smoothing > 0:
    #    y_true = y_true*(1-label_smoothing) + 0.5*label_smoothing
    # the sparse categorical crossent function will return a 
    # 1D tensor of length batch_size
    crossent = K.sparse_categorical_crossentropy(y_true_clipped, y_pred)
    # return the average over the unmasked values
    return K.sum(crossent*mask)/norm
            
            
        
def multilabel_distillation_loss(teacher, student, temp=1):
    """
    Loss function for knowledge distillation. Based on equation
    (2) from "Big Self-Supervised Models are Strong Semi-Supervised
    Learners" by Chen et al, and adapted for our multilabel 
    binary classifier setup.
    
    NOTE check to make sure teacher and student shapes are compatible-
    got some weird results when they weren't.
    
    :teacher: output from teacher model
    :student: output from student model
    """
    teacher = tf.nn.sigmoid((2*teacher-1)/temp)
    student = tf.nn.sigmoid((2*student-1)/temp)
    
    return -1*tf.reduce_mean(teacher*tf.math.log(student+K.epsilon()) + \
            (1-teacher)*tf.math.log(1-student+K.epsilon()))

            
            