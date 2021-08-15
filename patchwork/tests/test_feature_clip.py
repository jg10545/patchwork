# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf

from patchwork.feature._clip import _add_1d_residual_attention_block
from patchwork.feature._clip import build_bagofwords_transformer
from patchwork.feature._clip import get_vocab
from patchwork.feature._clip import build_text_encoder
    



def test_add_1d_residual_attention_block():
    d = 23
    inpt = tf.keras.layers.Input((d,))
    outpt = _add_1d_residual_attention_block(inpt, 11, 7)
    assert len(outpt.shape) == 2
    assert outpt.shape[1] == d


def test_build_bagofwords_transformer():
    d = 23
    project_to = 11
    num_heads = 3
    
    for num_layers in [2,3]:
        tform = build_bagofwords_transformer(d, project_to, num_layers, num_heads)
        # check the number of layers- one for input, one for initial projection,
        # six per attention block
        assert len(tform.layers) == 2 + 6*num_layers
        
        
def test_get_vocab():
    corpus = ["this is a test", "this is also a test", "my dog has fleas"]
    vocab = get_vocab(corpus, min_df=2)
    assert isinstance(vocab, list)
    assert len(vocab) == 3
    
def test_build_text_encoder():
    vocab = ["my", "dog", "has", "fleas", "habedashery"]
    teststr = np.array([["My dog, has fleas!"]])
    
    encoder = build_text_encoder(vocab, 13, 5, 3, None, 7)
    outpt = encoder(teststr).numpy()
    assert outpt.shape == (1,7)