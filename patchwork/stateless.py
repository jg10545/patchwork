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

from patchwork._sample import stratified_sample, find_labeled_indices, find_unlabeled
from patchwork._sample import find_excluded_indices, PROTECTED_COLUMN_NAMES
from patchwork._badge import KPlusPlusSampler, _build_output_gradient_function
from patchwork._losses import masked_binary_crossentropy
from patchwork._labeler import pick_indices
from patchwork._util import shannon_entropy


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



def _labels_to_dataframe(labels, classes=None):
    """
    Input labels as a list of dictionaries; return same information as
    a pandas dataframe.
    
    :classes: if a list of strings is passed here, will prune the columns
        to just these categories
    """
    # convert to dataframe
    df = pd.DataFrame(labels)
    
    # if exclude and validation not present, add them
    for c in ["exclude", "validation"]:
        if c not in df.columns:
            df[c] = False
            
    if "filepath" not in df.columns:
        df["filepath"] = ""
            
    # prune if necessary
    if classes is not None:
        for c in df.columns:
            if c not in classes+PROTECTED_COLUMN_NAMES:
                df = df.drop(c, axis=1)
    # make sure columns are in the same order
    if classes is not None:
        df = df.reindex(columns=["filepath", "exclude", 
                                 "validation"]+classes)     
    return df

        

def train(features, labels, classes, training_steps=1000, 
          batch_size=32, learning_rate=1e-3, **model_kwargs):
    """
    Hey now, you're an all-star. Get your train on.
    
    
    :features: (num_samples, input_dim) array of feature vectors
    :training_steps: how many steps to train for
    :classes: list of strings; names of categories to include in model
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
    df = _labels_to_dataframe(labels, classes)
    
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



def get_indices_of_tiles_in_predicted_class(features, model, category_index, 
                                            threshold=0.5):
    """
    :features: (num_samples, input_dim) array of feature vectors
    :model: trained Keras classifier model
    :category_index: integer index of category to query
    :threshold: float between 0 and 1; minimum probability assessed by
        classifier
    """
    predictions = model.predict(features)
    category_predictions = predictions[:,category_index]
    return np.arange(features.shape[0])[category_predictions >= threshold]




def sample_random(labels, max_to_return=None):
    """
    Generate a random sample of indices
    
    :labels: list of dictionaries containing labels
    :max_to_return: if not None; max number of indices to return
    """
    N = len(labels)
    if max_to_return is None:
        max_to_return = N
    # create a list of unlabeled indices    
    df = _labels_to_dataframe(labels)
    labeled = list(find_labeled_indices(df))
    indices_to_sample_from = [n for n in range(N) if n not in labeled]
    # update num_to_return in case not many unlabeled are left
    max_to_return = min(max_to_return, len(indices_to_sample_from))
    
    return np.random.choice(indices_to_sample_from,size=max_to_return,
                            replace=False)

def sample_uncertainty(labels, features, model, max_to_return=None):
    """
    Return indices sorted by decreasing entropy.
    
    :features: (num_samples, input_dim) array of feature vectors
    :model: trained Keras classifier model
    :max_to_return: if not None; max number of indices to return
    """  
    N = features.shape[0]
    # get model predictions
    predictions = model.predict(features)
    # compute entropies for each prediction
    entropy = shannon_entropy(predictions)
    
    # create a list of unlabeled indices    
    df = _labels_to_dataframe(labels)
    labeled = list(find_labeled_indices(df))
    
    # order by decreasing entropy
    ordering = entropy.argsort()[::-1]
    ordered_indices = np.arange(N)[ordering]
    # prune out labeled indices
    ordered_indices = [i for i in ordered_indices if i not in labeled]
    # and clip list if necessary
    if max_to_return is not None:
        ordered_indices = ordered_indices[:max_to_return]
    return ordered_indices



def sample_diversity(labels, features, model, max_to_return=None):
    """
    Return indices sorted by decreasing entropy.
    
    :features: (num_samples, input_dim) array of feature vectors
    :model: trained Keras classifier model
    :max_to_return: if not None; max number of indices to return
    """  
    N = features.shape[0]
    if max_to_return is None:
        max_to_return = N
        
    # compute badge embeddings- define a tf.function for it
    compute_output_gradients = _build_output_gradient_function(model)
    # then run that function across all the images.
    output_gradients = tf.map_fn(compute_output_gradients, features).numpy()
    
    # figure out which indices are yet labeled
    df = _labels_to_dataframe(labels)
    labeled = list(find_labeled_indices(df))
    # update max_to_return in case there aren't very many 
    # unlabeled indices left
    max_to_return = min(max_to_return, N-len(labeled))

    # initialize a K++ sampler
    badge_sampler = KPlusPlusSampler(output_gradients, indices=labeled)
    return badge_sampler(max_to_return)






