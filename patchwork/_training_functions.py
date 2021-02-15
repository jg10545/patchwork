# -*- coding: utf-8 -*-
import tensorflow as tf
import tensorflow.keras.backend as K

from patchwork._losses import entropy_loss
from patchwork._util import compute_l2_loss
from patchwork.feature._moco import exponential_model_update

def build_training_function(loss_fn, opt, fine_tuning, output, feature_extractor=None, lam=0, 
                            tau=0.95, weight_decay=0):
    """
    Generate a tensorflow function for training the model.
    
    :loss_fn: training loss function
    :opt: Keras optimizer
    :fine_tuning: Keras model that maps pre-extracted features to semantic vectors
    :output: Keras model that maps semantic vectors to class predictions
    :feature_extractor: Keras model; fully-convolutional network for mapping raw
    :lam:
    :tau:
    """
    trainvars = fine_tuning.trainable_variables + output.trainable_variables
    # are the features precomputed or generated on the fly?
    
    
    def model(x, training=True):
        # helper function to apply both output models
        vectors = fine_tuning(x, training=training)
        return output(vectors, training=training)
    
    def build_mask(x):
        # helper to generate mask for FixMatch. slightly different from
        # the paper since we're doing sigmoid multilabel learning
        confident_high = x >= tau
        confident_low = x <= (1-tau)
        mask = tf.math.logical_or(confident_high, confident_low)
        return tf.cast(mask, tf.float32)
    
    @tf.function
    def training_step(x, y, x_unlab_wk=None, x_unlab_str=None):
        # for dynamically-computed features, run images through the
        # feature extractor
        if feature_extractor is not None:
            x = feature_extractor(x)
            if lam > 0:
                x_unlab_wk = feature_extractor(x_unlab_wk)
                x_unlab_str = feature_extractor(x_unlab_str)
            
        with tf.GradientTape() as tape:
            # compute outputs for training data
            y_pred = model(x, True)
            
            # loss function between labels and predictions
            training_loss = loss_fn(y, y_pred)
            
            if weight_decay > 0:
                training_loss += weight_decay*compute_l2_loss(fine_tuning)
            
            total_loss = training_loss
            
            # semi-supervised case- loss function for unlabeled data
            # entropy regularization
            if lam > 0:
                with tape.stop_recording():
                    # GENERATE FIXMATCH PSEUDOLABELS
                    # make predictions on the weakly-augmented batch
                    unlab_preds = model(x_unlab_wk, False)
                    # round predictions to pseudolabels
                    pseudolabels = tf.cast(unlab_preds > 0.5, 
                                           tf.float32)
                    # also compute a mask from the predictions,
                    # since we only incorporate high-confidence cases,
                    # compute a mask that's 1 every place that's close
                    # to 1 or 0
                    mask = build_mask(unlab_preds)
                
                # MAKE PREDICTIONS FROM STRONG AUGMENTATION
                str_preds = model(x_unlab_str, True)
                crossent_tensor = K.binary_crossentropy(pseudolabels,
                                                        str_preds)
                fixmatch_loss = tf.reduce_mean(mask*crossent_tensor)
                total_loss += lam*fixmatch_loss
            else:
                fixmatch_loss = 0
                
        # compute and apply gradients
        gradients = tape.gradient(total_loss, trainvars)
        opt.apply_gradients(zip(gradients, trainvars))

        return training_loss, fixmatch_loss
    return training_step
