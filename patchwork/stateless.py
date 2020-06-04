# -*- coding: utf-8 -*-
"""

                stateless.py
                
                
Stateless training and active learning wrapper functions
for integrating with an app.


"""
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras.backend as K
import pickle

from patchwork._sample import stratified_sample, find_labeled_indices, find_excluded_indices
from patchwork._badge import KPlusPlusSampler, _build_output_gradient_function
from patchwork._losses import masked_binary_crossentropy
from patchwork._labeler import pick_indices


def pickle_keras_model(model, file):
    """
    Dump a pickle object of a Keras model. Directly pickling
    Keras models returns a "TypeError: can't pickle _thread.RLock
    objects" error. This function breaks the model into two
    components:
        -a JSON definition of the model structure
        -a list of numpy arrays specifying the weights
        
    The (structure, weights) tuple is then pickled.
    """
    model_tuple = (model.to_json(), model.get_weights())
    pickle.dump(model_tuple, file)
    
def load_model_from_pickle(file):
    """
    Inverse of pickle_keras_model(). Input a file object, 
    load the pickled tuple, construct model structure from
    JSON and set the saved weights.
    
    Returns an uncompiled Keras model.
    """
    model_spec, model_weights = pickle.load(file)
    model = tf.keras.models.model_from_json(model_spec)
    model.set_weights(model_weights)
    return model


def _build_model(input_dim, num_classes, num_hidden_layers=0, 
                 hidden_dimension=128,
                 normalize_inputs=False, dropout=0):
    """
    Macro to generate a Keras classification model
    """
    inpt = tf.keras.layers.Input((input_dim))
    net = inpt
    
    # if we're normalizing inputs:
    if normalize_inputs:
        norm = tf.keras.layers.Lambda(lambda x:K.l2_normalize(x,axis=1))
        net = norm(net)
        
    # for each hidden layer
    for _ in range(num_hidden_layers):
        if dropout > 0:
            net = tf.keras.layers.Dropout(dropout)(net)
        net = tf.keras.layers.Dense(hidden_dimension, activation="relu")(net)
        
    # final layer
    if dropout > 0:
        net = tf.keras.layers.Dropout(dropout)(net)
    net = tf.keras.layers.Dense(num_classes, activation="relu")(net)
    
    return tf.keras.Model(inpt, net)

def _build_training_dataset(features, df, num_classes, num_samples, 
                            batch_size):
    indices, labels = stratified_sample(df, num_samples, 
                                            return_indices=True)
    ds = tf.data.Dataset.from_tensor_slices((features[indices], labels))
    ds = ds.batch(batch_size)
    ds = ds.prefetch(1)
    return ds

def train(features, classes, labels, training_steps=1000, 
          batch_size=16, learning_rate=1e-3, **model_kwargs):
    """
    Hey now, you're an all-star. Get your train on.
    
    
    :features: (num_samples, input_dim) array of feature vectors
    :classes: list of strings; names of categories to include in model
    :training_steps: how many steps to train for
    :batch_size: number of examples per batch
    :learning_rate: learning rate for Adam optimizer
    :model_kwargs: keyword arguments for model construction
    
    Returns
    :model: trained Keras model
    :training_loss: array of length (training_steps) giving the
        loss function value at each training step
    :validation_metrics: dictionary of validation metrics NOT YET
        IMPLEMENTED
    """
    # convert labels to a dataframe
    df = pd.DataFrame(labels)
    print("incorporate classes arg")
    
    num_classes = len(classes)
    input_dim = features.shape[1]
    
    # create a model and optimizer
    model = _build_model(input_dim, num_classes, **model_kwargs)
    opt = tf.keras.optimizers.Adam(learning_rate)
    
    # build a dataset for training
    num_samples = training_steps*batch_size
    ds = _build_training_dataset(features, df, num_classes, num_samples, 
                                 batch_size)
        
    # train the model, recording loss at each step
    training_loss = []
    @tf.function
    def training_step(x,y):
        with tf.GradientTape() as tape:
            pred = model(x, training=True)
            loss = masked_binary_crossentropy(y, pred)
        variables = model.trainable_variables
        grads = tape.gradient(loss, variables)
        opt.apply_gradients(zip(grads, variables))
        return loss
        
    for x, y in ds:
        training_loss.append(training_step(x,y).numpy())
        
    return model, np.array(training_loss), {}


def predict(features, model):
    """
    :features: (num_samples, input_dim) array of feature vectors
    :model: a trained Keras model
    
    Returns a (num_samples, num_classes) array of multiclass sigmoid
        probabilities
    """
    return model.predict(features)



















