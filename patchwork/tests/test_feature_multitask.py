# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import tensorflow as tf

from patchwork.feature._multitask import _encode_classes, _dataframe_to_classes
from patchwork.feature._multitask import _assemble_full_network


def test_encode_classes():
    train = pd.Series(["foo", "bar", "bar", np.nan])
    val = pd.Series(["bar", "bar"])
    
    train_ind, val_ind, classes = _encode_classes(train, val)
    
    assert len(classes) == 2
    for c in ["foo", "bar"]:
        assert c in classes
    assert train_ind.shape[0] == len(train)
    assert val_ind.shape[0] == len(val)
    assert -1 in train_ind
    assert -1 not in val_ind
    
    
def test_dataframe_to_classes():
    train = pd.DataFrame({
        "filepath":["foo.png", "bar.png", "foobar.png"],
        "class0":["a", "b", "c"],
        "class1":["x", "y", np.nan]
    })
    val = pd.DataFrame({
        "filepath":["foo1.png", "bar2.png"],
        "class0":["a", "b"],
        "class1":["x", "y"]
    })
    outdict, class_dict = _dataframe_to_classes(train, val,
                                    ["class0", "class1"])
    
    assert len(outdict["train_files"]) == len(train)
    assert len(outdict["val_files"]) == len(val)
    
    assert outdict["train_indices"].shape == (len(train),2)
    assert outdict["val_indices"].shape == (len(val),2)
    assert -1 in outdict["train_indices"]
    assert -1 not in outdict["val_indices"]
    
    
def test_assemble_full_network():
    inpt = tf.keras.layers.Input((None, None, 3))
    net = tf.keras.layers.Conv2D(5, 3)(inpt)
    fcn = tf.keras.Model(inpt, net)
    
    task_dimensions = [2,3,4]
    model_dict, trainvars = _assemble_full_network(fcn,
                                   task_dimensions,
                                   shared_layers=[3,5],
                                   task_layers=[7,"p",11],
                                   train_fcn=False,
                                   global_pooling="max")
    
    assert model_dict["fcn"] is fcn
    assert len(model_dict["full"].outputs) == 3
    for o,d in zip(model_dict["full"].outputs, task_dimensions):
        assert o.shape[-1] == d
    assert isinstance(trainvars, list)