# -*- coding: utf-8 -*-
import tensorflow as tf


class TransformerBlock(tf.keras.layers.Layer):
    """
    Transfomer Block as a keras layer.
    
    from here: https://keras.io/examples/nlp/text_classification_with_transformer/
    """
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential(
            [tf.keras.layers.Dense(ff_dim, activation="relu"), tf.keras.layers.Dense(embed_dim),]
        )
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, inputs, training=True):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)
    
    
class TokenAndPositionEmbedding(tf.keras.layers.Layer):
    """
    keras layer for token + positional embeddings.
    
    from here: https://keras.io/examples/nlp/text_classification_with_transformer/
    
    Note that this uses learned positional embeddings instead of sinusoids. From Attention
    is All You Need: "We also experimented with using learned positional embeddings [9] 
    instead, and found that the two versions produced nearly identical results"
    """
    def __init__(self, maxlen, vocab_size, embed_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_emb = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = tf.keras.layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions
    
    
def build_text_transformer(vocab_size, maxlen, embed_dim=512, num_layers=12, num_heads=8, 
                           ff_dim=2048, final_projection=False):
    """
    Assemble TransformerBlock and TokenAndPositionEmbedding layers into a text transformer.
    
    :vocab_size: int; number of (BPE) tokens in the vocabulary
    :maxlen: int; length of text sequences (BPE tokens, not raw string lengths). preprocessing should
        pad/clip to this value
    :embed_dim: int; embedding dimension for tokens and transformers
    :num_layers: int; number of transformer blocks
    :num_heads: int; number of heads in each transformer block
    :ff_dim: int; dimension of internal feed-forward layers inside transformer blocks (2048 was value from
        Attention is All You Need)
    :final_projection: if an integer; pool and project final transformer block to this dimension
    """
    inpt = tf.keras.layers.Input((maxlen,))
    x = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)(inpt)
        
    for n in range(num_layers):
        x = TransformerBlock(embed_dim, num_heads, ff_dim)(x)
    if final_projection:
        print(x.shape)
        x = tf.keras.layers.Lambda(lambda y: y[:,-1,:])(x)
        #x = tf.keras.layers.GlobalAvgPool1D(data_format='channels_last')(x)
        print(x.shape)
        x = tf.keras.layers.Dense(final_projection)(x)
        print(x.shape)
    
    return tf.keras.Model(inpt, x)