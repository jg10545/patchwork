# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
import sklearn.feature_extraction


def _add_1d_residual_attention_block(inpt, num_heads, key_dim, attention_axes=[1]):
    """
    Add a residual attention block to a Keras model. patterned after this:
    
    https://github.com/openai/CLIP/blob/fa56f2525191a013533338f137aab59ac36d8c26/clip/model.py#L167
    """
    d_model = inpt.shape[-1]
    x = tf.keras.layers.LayerNormalization()(inpt)
    x = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim,
                                          attention_axes=attention_axes)(x,x)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.Dense(4*d_model)(x)
    x = tf.keras.layers.Activation(tf.nn.gelu)(x)
    x = tf.keras.layers.Dense(d_model)(x)
    return x


def build_bagofwords_transformer(d, project_to=512, num_layers=12, num_heads=12, key_dim=None, final_projection=False):
    """
    Build a simple transformer for text encoded as a bag-of-words. Project to a lower dimension, then
    add residual attention blocks and(optionally) a final linear projection/
    """
    if key_dim is None:
        key_dim = project_to
        
    inpt = tf.keras.layers.Input((d,))
    x = tf.keras.layers.Dense(project_to, activation="relu")(inpt)
    for n in range(num_layers):
        x = _add_1d_residual_attention_block(x, num_heads, key_dim)
    if final_projection:
        x = tf.keras.layers.Dense(final_projection)(x)
    
    return tf.keras.Model(inpt, x)


def get_vocab(corpus, max_features=5000, min_df=1, max_df=1., **kwargs):
    """
    """
    vec = sklearn.feature_extraction.text.CountVectorizer(max_features=max_features, min_df=min_df, 
                                                          max_df=max_df, **kwargs)
    vec.fit(corpus)
    vocab = list(vec.vocabulary_.keys())
    return vocab

def multihot_encode(x, depth):
    onehot = tf.one_hot(x, depth=depth)
    return tf.reduce_sum(onehot, 1)


def build_text_encoder(vocab, project_to=512, num_layers=12, num_heads=12, key_dim=None, final_projection=False):
    """
    
    """
    V = len(vocab)
    vectorizer = tf.keras.layers.experimental.preprocessing.TextVectorization(vocabulary=vocab)
    encoder = tf.keras.layers.Lambda(lambda x: multihot_encode(x, V))
    tfm = build_bagofwords_transformer(V, project_to, num_layers, num_heads, key_dim, final_projection)
    
    inpt = tf.keras.layers.Input((), dtype=tf.string)
    x = vectorizer(inpt)
    x = encoder(x)
    x = tfm(x)
    return tf.keras.Model(inpt, x)  



def build_image_encoder(fcn, num_channels=3, output_dim=64):
    """
    NOT the full version used in OpenAI's paper- just a linear
    projection head after the global average pool, instead of
    a multi-head attention mechanism
    """
    inpt = tf.keras.layers.Input((None, None, 3))
    x = fcn(inpt)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(output_dim)(x)
    return tf.keras.Model(inpt, x)

def compute_nce_loss(img_embed, text_embed, temp=0.07):
    """
    Symmetrized NCE loss for paired image/text embeddings
    """
    N = img_embed.shape[0]
    img_norm = tf.nn.l2_normalize(img_embed, 1)
    text_norm = tf.nn.l2_normalize(text_embed, 1)
    # NOTE this is different from what's described in the paper- check 
    # pseudocode in figure 3
    logits1 = tf.matmul(img_norm, text_norm, transpose_b=True)/temp
    labels1 = tf.range(N)
    loss1 = tf.reduce_mean(
        tf.losses.sparse_categorical_crossentropy(labels1, logits1, from_logits=True))
    
    logits2 = tf.matmul(text_norm, img_norm, transpose_b=True)/temp
    labels2 = tf.range(N)
    loss2 = tf.reduce_mean(
        tf.losses.sparse_categorical_crossentropy(labels2, logits2, from_logits=True))
    return 0.5*(loss1 + loss2)

