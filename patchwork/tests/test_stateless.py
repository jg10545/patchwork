# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import tensorflow as tf

import patchwork.stateless as pws

def test_build_model():
    input_dim = 3
    num_classes = 5
    num_hidden_layers = 7
    hidden_dimension = 11
    normalize_inputs = False
    dropout = 0
    model = pws._build_model(input_dim, num_classes, num_hidden_layers,
                     hidden_dimension, normalize_inputs, dropout)
    
    assert isinstance(model, tf.keras.Model)
    assert len(model.layers) == num_hidden_layers+2
    assert model.output_shape == (None, num_classes)


def test_build_model_with_dropout_and_norm():
    input_dim = 3
    num_classes = 5
    num_hidden_layers = 7
    hidden_dimension = 11
    normalize_inputs = True
    dropout = 0.5
    model = pws._build_model(input_dim, num_classes, num_hidden_layers,
                     hidden_dimension, normalize_inputs, dropout)
    
    assert isinstance(model, tf.keras.Model)
    assert len(model.layers) == 2*num_hidden_layers+4
    assert model.output_shape == (None, num_classes)
    
    
def test_build_training_dataset():
    N = 2
    d = 5
    features = np.random.normal(0, 1, (N,d))
    
    df = pd.DataFrame({"filepath":["foo", "bar"],
                       "exclude":[False,False],
                       "validation":[False,False],
                       "class0":[1,0],
                       "class1":[0,1],
                       "class2":[1,1]})
    
    ds = pws._build_training_dataset(features, df, 3, 
                                     10, 16)
    
    assert isinstance(ds, tf.data.Dataset)
    for x,y in ds:
        break
    assert x.shape == (10,5)
    assert y.shape == (10,3)
    
    
    
def test_labels_to_dataframe():
    labels = [{"class0":0, "class1":1},
              {"class0":1, "class1":1},
              {"class0":0, "class1":0}]
    
    df = pws._labels_to_dataframe(labels)
    
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 3
    for c in ["filepath", "exclude", "validation", "class0", "class1"]:
        assert c in df.columns
        
        
def test_estimate_accuracy():
    tp = 5
    fp = 1
    tn = 5
    fn = 1
    
    acc = pws._estimate_accuracy(tp, fp, tn, fn)
    assert isinstance(acc, dict)
    assert acc["base_rate"] == 0.5
    assert acc["interval_low"] < acc["interval_high"]
        
def test_eval():
    N = 3
    d = 7
    features = np.random.normal(0,1,(N,d))
    df = pws._labels_to_dataframe([
        {"class0":0, "class1":1, "validation":True},
              {"class0":1, "class1":1, "validation":True},
              {"class0":0, "class1":0, "validation":True}
        ])
    classes = ["class0", "class1"]
    
    inpt = tf.keras.layers.Input((d,))
    outpt = tf.keras.layers.Dense(len(classes))(inpt)
    model = tf.keras.Model(inpt, outpt)
    
    acc = pws._eval(features, df, classes, model)
    assert isinstance(acc, dict)
    assert len(acc) == len(classes)
    assert len(acc["class0"]) == 5
        
        
        
        
        
        