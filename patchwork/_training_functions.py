# -*- coding: utf-8 -*-
import tensorflow as tf

from patchwork._losses import entropy_loss

def build_training_function(loss_fn, fine_tuning, output, feature_extractor=None, entropy_reg_weight=0):
    """
    Generate a tensorflow function for training the model.
    
    :loss_fn: training loss function
    :fine_tuning: Keras model that maps pre-extracted features to semantic vectors
    :output: Keras model that maps semantic vectors to class predictions
    :feature_extractor: Keras model; fully-convolutional network for mapping raw
    :entropy_reg_weight: float; weight for entropy regularization loss. 0 to disable
    """
    trainvars = fine_tuning.trainable_variables + output.trainable_variables
    # are the features precomputed or generated on the fly?
    
    @tf.function
    def training_step(x, y, opt, x_unlab=None):
        # for dynamically-computed features, run images through the
        # feature extractor
        if feature_extractor is not None:
            x = feature_extractor(x)
            
        with tf.GradientTape() as tape:
            # compute outputs for training data
            vectors = fine_tuning(x)
            y_pred = output(vectors)
            
            # loss function between labels and predictions
            training_loss = loss_fn(y, y_pred)
            
            # semi-supervised case- loss function for unlabeled data
            if entropy_reg_weight > 0:
                if feature_extractor is not None:
                    x_unlab = feature_extractor(x_unlab)
                vectors_ss = fine_tuning(x_unlab)
                pred_ss = output(vectors_ss)
                
                entropy_reg_loss = entropy_loss(pred_ss)
            else:
                entropy_reg_loss = 0.
            
            total_loss = training_loss + entropy_reg_weight*entropy_reg_loss
        # compute and apply gradients
        gradients = tape.gradient(total_loss, trainvars)
        opt.apply_gradients(zip(gradients, trainvars))
        return training_loss, entropy_reg_loss
    return training_step
