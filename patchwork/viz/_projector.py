"""

            _projector.py
            
Code to use with the tensorboard projector

"""
import numpy as np
import tensorflow as tf
from tensorboard.plugins import projector
from PIL import Image
import os

import patchwork
from patchwork._util import _load_img



def _make_sprite(imfiles, norm=1, num_channels=3, resize=(50,50)):
    """
    Input a 4D tensor, output a sprite image. 
    assumes your pictures are all the
    same size and square
    """
    num_sprites = len(imfiles)
    gridsize = np.int(np.ceil(np.sqrt(num_sprites)))
    sprite_arr = np.stack([
        _load_img(f, norm, num_channels, resize)
        for f in imfiles
    ])#.astype(np.uint8)

    output = np.zeros((resize[0]*gridsize, resize[1]*gridsize, 3), 
                      dtype=np.uint8)
    
    for i in range(num_sprites):
        col = i // gridsize
        row = i % gridsize
        output[resize[0]*col:resize[0]*(col+1), 
               resize[1]*row:resize[1]*(row+1),:] = sprite_arr[i,:,:,:3]
    img = Image.fromarray(output)
    img = img.resize((resize[0]*gridsize, resize[1]*gridsize))
    
    return img


def generate_embedding_op(vecs, spritesize=50):
    """
    Add a Variable to the graph to store embeddings for some 
    of your data points.
    
    :vecs: 2D numpy array of feature vectors
    
    Returns
    :store_embeddings: graph op to update your embedding vector
    :config: projector config
    """
    embed_dummy = tf.get_variable("dense_embeddings", shape=vecs.shape,
                              initializer=tf.initializers.random_uniform())
    store_embeddings = tf.assign(embed_dummy, tf.constant(vecs))

    config = projector.ProjectorConfig()
    embedding = config.embeddings.add()
    embedding.tensor_name = embed_dummy.name

    embedding.sprite.image_path = "sprites.png" 
    embedding.sprite.single_image_dim.extend([spritesize, spritesize])
    
    return store_embeddings, config


def build_tensorboard_projections(feature_extractor, filepaths, logdir,
                                pooling="max", spritesize=50, norm=1,
                                 num_channels=3, imshape=(256,256), batch_size=256):
    """
    Run a list of images through a convolutional feature extractor,
    and save the results in a format compatible with the Tensorboard
    projector tool.
    
    :feature_extractor:
    :filepaths:
    :logdir:
    :pooling:
    :spritesize:
    :norm:
    :num_channels:
    :imshape:
    :batch_size:
    """
    # compute embeddings
    ds, steps = patchwork._loaders.dataset(filepaths, imshape=imshape,
                                   num_channels=num_channels,
                                   norm=norm, batch_size=batch_size)
    feature_vecs = feature_extractor.predict(ds, steps=steps)
    if pooling == "max":
        feature_vecs = feature_vecs.max(axis=1).max(axis=1)
    else:
        assert False, "not yet implemented"
        
    # build sprites
    sprite = _make_sprite(filepaths, norm=norm, 
                          num_channels=num_channels, 
                          resize=(spritesize, spritesize))
    sprite.save(os.path.join(logdir, "sprites.png"))
    
    # build graph and run session
    embed_op, config = generate_embedding_op(feature_vecs, spritesize)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        # object to save training summaries
        writer = tf.summary.FileWriter(logdir, sess.graph, 
                                   flush_secs=5)
        # record visualization metadata
        projector.visualize_embeddings(writer, config)
        sess.run(tf.global_variables_initializer())
        _ = sess.run(embed_op)
    
        saver.save(sess, os.path.join(logdir, "model.ckpt"), 1)
