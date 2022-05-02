# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import tensorflow as tf
import scipy.sparse

from patchwork._labelprop import _get_weighted_adjacency_matrix, _propagate_labels
import patchwork as pw

def test_get_weighted_adjacency_matrix():
    N = 100
    d = 10

    features = np.random.normal(0, 1, (N,d)).astype(np.float32)
    features = tf.nn.l2_normalize(features, 1).numpy()
    
    W_norm = _get_weighted_adjacency_matrix(features, n_neighbors=5)
    assert isinstance(W_norm, scipy.sparse.csr.csr_matrix)
    assert W_norm.shape == (N,N)
    assert round(W_norm.min(),2) == 0
    assert round(W_norm.max(),2) <= 1

def test_propagate_labels():
    # build out some random data
    N = 100
    d = 10
    features = np.random.normal(0, 1, (N,d)).astype(np.float32)
    features = tf.nn.l2_normalize(features, 1).numpy()
    classes = ["foo", "bar"]
    filepaths = [f"{i}.jpg" for i in range(N)]
    df = pw.prep_label_dataframe(filepaths, classes)
    for i in range(10):
        df.loc[i, "foo"] = np.random.choice([0,1])
        df.loc[i, "bar"] = np.random.choice([0,1])
        df.loc[i, "validation"] = np.random.choice([True, False])
        
    W_norm = _get_weighted_adjacency_matrix(features, n_neighbors=5)
    pred_df = _propagate_labels(df, W_norm)
    
    assert isinstance(pred_df, pd.DataFrame)
    assert len(pred_df) == N
    assert "foo" in pred_df.columns
    assert "bar" in pred_df.columns
    

