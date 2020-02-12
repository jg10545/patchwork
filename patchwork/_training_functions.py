# -*- coding: utf-8 -*-
import tensorflow as tf

from patchwork._losses import entropy_loss

def build_training_function(loss_fn, fine_tuning, output, feature_extractor=None,
                            entropy_reg_weight=0, mean_teacher_alpha=0,
                            teacher_finetune=None, teacher_output=None):
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
            vectors = fine_tuning(x, training=True)
            y_pred = output(vectors, training=True)
            
            # loss function between labels and predictions
            training_loss = loss_fn(y, y_pred)
            
            # semi-supervised case- loss function for unlabeled data
            # entropy regularization
            if entropy_reg_weight > 0:
                if feature_extractor is not None:
                    x_unlab = feature_extractor(x_unlab)
                vectors_ss = fine_tuning(x_unlab, training=True)
                pred_ss = output(vectors_ss, training=True)
                
                entropy_reg_loss = entropy_loss(pred_ss)
            # mean teacher
            elif mean_teacher_alpha > 0:
                if feature_extractor is not None:
                    x_unlab = feature_extractor(x_unlab)
                vectors_ss = fine_tuning(x_unlab, training=True)
                pred_ss = output(vectors_ss, training=True)
                
                teach_vector_ss = teacher_finetune(x_unlab, training=True)
                teach_pred_ss = teacher_output(teach_vector_ss, training=True)
                
                entropy_reg_loss = tf.reduce_sum((pred_ss-teach_pred_ss)**2)
            else:
                entropy_reg_loss = 0.
            
            total_loss = training_loss + entropy_reg_weight*entropy_reg_loss
        # compute and apply gradients
        gradients = tape.gradient(total_loss, trainvars)
        opt.apply_gradients(zip(gradients, trainvars))
        
        if mean_teacher_alpha > 0:
            from patchwork.feature._moco import exponential_model_update
            _ = exponential_model_update(teacher_finetune, fine_tuning,
                                         mean_teacher_alpha)
            _ = exponential_model_update(teacher_output, output,
                                         mean_teacher_alpha)
        return training_loss, entropy_reg_loss
    return training_step
