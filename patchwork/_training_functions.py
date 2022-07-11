# -*- coding: utf-8 -*-
import tensorflow as tf
import tensorflow.keras.backend as K

#from patchwork._losses import entropy_loss
from patchwork._util import compute_l2_loss
from patchwork.feature._moco import exponential_model_update, copy_model
from patchwork._domain import _compute_mmd_loss

def build_training_function(loss_fn, opt, fine_tuning, output, feature_extractor=None, lam=0, 
                            tau=0.95, weight_decay=0, finetune=False, domain_weight=0, 
                            num_domains=0):
    """
    Generate a tensorflow function for training the model.
    
    :loss_fn: training loss function
    :opt: Keras optimizer
    :fine_tuning: Keras model that maps pre-extracted features to semantic vectors
    :output: Keras model that maps semantic vectors to class predictions
    :feature_extractor: Keras model; fully-convolutional network for mapping raw
    :lam:
    :tau:
    :weight_decay:
    :finetune:
    :domain_weight:
    :num_domains:
    """
    trainvars = fine_tuning.trainable_variables + output.trainable_variables
    if finetune&(feature_extractor is not None):
        trainvars += feature_extractor.trainable_variables
    
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
    
    def training_step(x, y, x_unlab_wk=None, x_unlab_str=None, domain_labels=None):
        # build pseudolabels and mask outside gradienttape
        if lam > 0:
            assert (x_unlab_wk is not None)&(x_unlab_str is not None)
            if (feature_extractor is not None):
                x_unlab_wk = feature_extractor(x_unlab_wk)
            preds_wk = model(x_unlab_wk)
            # round weak predictions to pseudolabels
            pseudolabels = tf.cast(preds_wk > 0.5, 
                                           tf.float32)
            # also compute a mask from the predictions,
            # since we only incorporate high-confidence cases,
            # compute a mask that's 1 every place that's close
            # to 1 or 0
            mask = build_mask(preds_wk)
        
        # for dynamically-computed features, run images through the
        # feature extractor
        if (feature_extractor is not None)&(finetune == False):
            x = feature_extractor(x)
            if x_unlab_str is not None:
                x_unlab_str = feature_extractor(x_unlab_str)
            
        with tf.GradientTape() as tape:
            # run images through the feature extractor if we're fine-tuning
            if (feature_extractor is not None)&finetune:
                x = feature_extractor(x, training=True)
                if x_unlab_str is not None:
                    x_unlab_str = feature_extractor(x_unlab_str, training=True)
            
            # compute outputs for training data
            y_pred = model(x, True)
            
            # semi-supervised case- loss function for unlabeled data
            if lam > 0:
                assert domain_weight==0, "fixmatch + domain confusion not implemented"
                preds_str = model(x_unlab_str, True)

                crossent_tensor = K.binary_crossentropy(pseudolabels,
                                                        preds_str)
                semisup_loss = tf.reduce_mean(mask*crossent_tensor)
                semisup_weight = lam
            elif domain_weight > 0:
                if feature_extractor is not None:
                    x_unlab_wk = feature_extractor(x_unlab_wk, training=True)
                features = fine_tuning(x_unlab_wk, training=True)
                semisup_loss = _compute_mmd_loss(features, domain_labels, num_domains)
                semisup_weight = domain_weight
            else:
                semisup_loss = 0
                semisup_weight = 0
            
            # supervised loss function between labels and predictions
            training_loss = loss_fn(y, y_pred)
            
            if (weight_decay > 0)&(len(fine_tuning.trainable_variables)>0):
                l2_loss = compute_l2_loss(fine_tuning)
            else:
                l2_loss = 0
                
            if (weight_decay > 0)&(feature_extractor is not None)&finetune:
                l2_loss += compute_l2_loss(feature_extractor)
            
            total_loss = training_loss + weight_decay*l2_loss + semisup_weight*semisup_loss
                
        # compute and apply gradients
        gradients = tape.gradient(total_loss, trainvars)
        opt.apply_gradients(zip(gradients, trainvars))

        return training_loss, semisup_loss
    return training_step
