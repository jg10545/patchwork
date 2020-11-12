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
    assert acc["base_rate"] == "0.500"
    #assert acc["interval_low"] < acc["interval_high"]
        
def test_eval():
    N = 6
    d = 7
    features = np.random.normal(0,1,(N,d))
    df = pws._labels_to_dataframe([
        {"class0":0, "class1":1, "validation":True},
              {"class0":1, "class1":1, "validation":True},
              {"class0":0, "class1":0, "validation":True},
              {"class0":0, "class1":1, "validation":False},
              {"class0":1, "class1":1, "validation":False},
              {"class0":0, "class1":0, "validation":False}
        ])
    classes = ["class0", "class1"]
    
    inpt = tf.keras.layers.Input((d,))
    outpt = tf.keras.layers.Dense(len(classes))(inpt)
    model = tf.keras.Model(inpt, outpt)
    
    acc = pws._eval(features, df, classes, model)
    assert isinstance(acc, dict)
    assert len(acc) == len(classes)
    # check that all the validation metrics are there
    for k in ["accuracy", "base_rate", "prob_above_base_rate", "auc"]:
        assert k in acc["class0"]
    # also check to see we've got class predictions
    assert "predictions" in acc["class0"]
    for k in ["validation_positive", "validation_negative",
              "training_positive", "training_negative"]:
        assert k in acc["class0"]["predictions"]
        assert isinstance(acc["class0"]["predictions"][k], list)
        assert isinstance(acc["class0"]["predictions"][k][0], float)
        
         
def test_eval_with_partially_missing_val_labels():
    N = 6
    d = 7
    features = np.random.normal(0,1,(N,d))
    df = pws._labels_to_dataframe([
        {"class0":0, "validation":True},
              {"class0":1, "class1":1, "validation":True},
              {"class1":0, "validation":True},
              {"class0":0, "class1":1, "validation":False},
              {"class0":1, "class1":1, "validation":False},
              {"class0":0, "class1":0, "validation":False}
        ])
    classes = ["class0", "class1"]
    
    inpt = tf.keras.layers.Input((d,))
    outpt = tf.keras.layers.Dense(len(classes))(inpt)
    model = tf.keras.Model(inpt, outpt)
    
    acc = pws._eval(features, df, classes, model)
    assert isinstance(acc, dict)
    assert len(acc) == len(classes)
    # check that all the validation metrics are there
    for k in ["accuracy", "base_rate", "prob_above_base_rate", "auc"]:
        assert k in acc["class0"]
    # also check to see we've got class predictions
    assert "predictions" in acc["class0"]
    for k in ["validation_positive", "validation_negative",
              "training_positive", "training_negative"]:
        assert k in acc["class0"]["predictions"]
        assert isinstance(acc["class0"]["predictions"][k], list)
        assert isinstance(acc["class0"]["predictions"][k][0], float) 
        
        
def test_train_without_validation():
    N = 3
    d = 7
    features = np.random.normal(0,1,(N,d))
    labels = [
        {"class0":0, "class1":1},
              {"class0":1, "class1":1},
              {"class0":0, "class1":0}
        ]
    classes = ["class0", "class1"]
    model, loss, acc = pws.train(features, labels, classes,
                                 training_steps=1, batch_size=10,
                                 num_hidden_layers=0)
    
    
    assert isinstance(model, tf.keras.Model)
    assert isinstance(loss, np.ndarray)
    assert isinstance(acc, dict)
    assert len(loss) == 1
    assert len(acc) == len(classes)
    # nothing in class accuracy dicts because no validation
    assert len(acc["class0"]) == 0
        
        
        
def test_train_with_validation():
    N = 6
    d = 7
    features = np.random.normal(0,1,(N,d))
    labels = [
        {"class0":0, "class1":1},
              {"class0":1, "class1":1},
              {"class0":0, "class1":0},
              {"class0":0, "class1":1, "validation":True},
              {"class0":1, "class1":1, "validation":True},
              {"class0":0, "class1":0, "validation":True}
        ]
    classes = ["class0", "class1"]
    model, loss, acc = pws.train(features, labels, classes,
                                 training_steps=1, batch_size=10,
                                 num_hidden_layers=0)
    
    
    assert isinstance(model, tf.keras.Model)
    assert isinstance(loss, np.ndarray)
    assert isinstance(acc, dict)
    assert len(loss) == 1
    assert len(acc) == len(classes)
    assert "base_rate" in acc["class0"]
          
def test_train_with_focal_loss():
    N = 3
    d = 7
    features = np.random.normal(0,1,(N,d))
    labels = [
        {"class0":0, "class1":1},
              {"class0":1, "class1":1},
              {"class0":0, "class1":0}
        ]
    classes = ["class0", "class1"]
    model, loss, acc = pws.train(features, labels, classes,
                                 training_steps=1, batch_size=10,
                                 num_hidden_layers=0,
                                 focal_loss=True)
    
    
    assert isinstance(model, tf.keras.Model)
    assert isinstance(loss, np.ndarray)
    assert isinstance(acc, dict)
    assert len(loss) == 1
    assert len(acc) == len(classes)
    # nothing in class accuracy dicts because no validation
    assert len(acc["class0"]) == 0
    
def test_get_indices_of_tiles_in_predicted_class():
    N = 3
    d = 7
    features = np.random.normal(0,1,(N,d))
    
    inpt = tf.keras.layers.Input((d,))
    outpt = tf.keras.layers.Dense(2)(inpt)
    model = tf.keras.Model(inpt, outpt)
    
    indices = pws.get_indices_of_tiles_in_predicted_class(features, model, 0, threshold=0)
    assert isinstance(indices, np.ndarray)
    assert (len(indices)==0) or (indices.max() < N)
    
    
def test_mlflow(mocker):
    class FakeClass():
        def __init__(self):
            pass
        
    run = FakeClass()
    run.info = FakeClass()
    run.info.run_id = "fake_run_id"
    
    mocker.patch("mlflow.set_tracking_uri", return_value=None)
    mocker.patch("mlflow.set_experiment", return_value=None)
    mocker.patch("mlflow.start_run", return_value=run)
    
    run_id = pws._mlflow("fake_server_uri", "fake_experiment_name")
    assert run_id == "fake_run_id"
    
def test_train_with_mlflow(mocker):
    N = 6
    d = 7
    features = np.random.normal(0,1,(N,d))
    labels = [
        {"class0":0, "class1":1},
              {"class0":1, "class1":1},
              {"class0":0, "class1":0},
              {"class0":0, "class1":1, "validation":True},
              {"class0":1, "class1":1, "validation":True},
              {"class0":0, "class1":0, "validation":True}
        ]
    classes = ["class0", "class1"]
    
    class FakeClass():
        def __init__(self):
            pass
        
    run = FakeClass()
    run.info = FakeClass()
    run.info.run_id = "fake_run_id"
    
    mocker.patch("mlflow.set_tracking_uri", return_value=None)
    mocker.patch("mlflow.set_experiment", return_value=None)
    mocker.patch("mlflow.start_run", return_value=run)
    mocker.patch("mlflow.log_params", return_value=None)
    mocker.patch("mlflow.log_metrics", return_value=None)
    
    model, loss, acc = pws.train(features, labels, classes,
                                 training_steps=1, batch_size=10,
                                 num_hidden_layers=0)
    
    
    assert isinstance(model, tf.keras.Model)
    assert isinstance(loss, np.ndarray)
    assert isinstance(acc, dict)
    assert len(loss) == 1
    assert len(acc) == len(classes)
    assert "base_rate" in acc["class0"]
          