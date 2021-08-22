# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf

from patchwork.feature._text_transformer import TransformerBlock, TokenAndPositionEmbedding
from patchwork.feature._text_transformer import build_text_transformer


def test_transformerblock():
    embed_dim = 5
    num_heads = 3
    ff_dim = 7
    seq_len = 11
    block = TransformerBlock(embed_dim, num_heads, ff_dim)
    outpt = block(np.zeros((1, 11, embed_dim), dtype=np.float32))
    assert outpt.shape == (1, seq_len, embed_dim)
    
def test_tokenandpositionembedding():
    maxlen = 11
    vocab_size = 7
    embed_dim = 5

    embed = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)
    inpt_seq = np.arange(maxlen).reshape(1,-1)%vocab_size
    outpt = embed(inpt_seq)
    assert outpt.shape == (1, maxlen, embed_dim)
    
def test_build_text_transformer_with_projection():
    maxlen = 11
    vocab_size = 13
    embed_dim = 17
    ff_dim = 23
    num_layers = 3
    num_heads = 5
    final_proj = 29
    tform = build_text_transformer(vocab_size, maxlen, embed_dim, num_layers, num_heads, ff_dim, final_proj)
    assert len(tform.layers) == 2 + num_layers + 2 # input, token embedding, transformer blocks, pooling, output projection
    assert tform.output_shape == (None, final_proj)
    
def test_build_text_transformer_without_projection():
    maxlen = 11
    vocab_size = 13
    embed_dim = 17
    ff_dim = 23
    num_layers = 3
    num_heads = 5
    tform = build_text_transformer(vocab_size, maxlen, embed_dim, num_layers, num_heads, ff_dim, False)
    assert len(tform.layers) == 2 + num_layers # input, token embedding, transformer blocks
    assert tform.output_shape == (None, maxlen, embed_dim)