"""

            _projector.py
            
Code to use with the tensorboard projector

"""
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorboard.plugins import projector
from PIL import Image
import os

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import patchwork as pw
#from patchwork._util import _load_img



def _make_sprite(images, spritesize=50):
    """
    Input a 4D tensor, output a sprite image. 
    assumes your pictures are all the
    same size and square
    """
    num_sprites = images.shape[0]
    imsize = images.shape[1]
    gridsize = np.int(np.ceil(np.sqrt(num_sprites)))
    output = np.zeros((imsize*gridsize, imsize*gridsize, 3), dtype=np.uint8)

    for i in range(num_sprites):
        col = i // gridsize
        row = i % gridsize
        output[imsize*col:imsize*(col+1), imsize*row:imsize*(row+1),:] = (255*images[i,:,:,:]).astype(np.uint8)
    img = Image.fromarray(output)
    img = img.resize((spritesize*gridsize, spritesize*gridsize))
    return img





def save_embeddings(fcn, labeldict, log_dir, proj_dim=64, 
                    sprite_size=50, **input_config):
    """
    """
    # load images into memory
    X = np.array(list(labeldict.keys()))

    ds = pw.loaders.dataset(X, shuffle=False, **input_config)[0]
    images = np.concatenate([x.numpy() for x in ds], axis=0)

    # ------ FEATURES ------
    # compute features and flatten
    features = fcn.predict(images)
    features = features.reshape(features.shape[0], -1)
    # reduce dimension with PCA
    if proj_dim > 0:
        features_scaled = StandardScaler().fit_transform(features)
        pca = PCA(min(proj_dim, features.shape[0]))
        features = pca.fit_transform(features_scaled)
        
        
    # ------ METADATA ------
    traintest = ["train"]*len(labeldict)
    for i in range(len(traintest)):
        if i % 3 == 0:
            traintest[i] = "test"
    metadata = pd.DataFrame({"label":list(labeldict.values()),
                             "traintest":traintest})
    metadata.to_csv(os.path.join(log_dir, "metadata.tsv"), sep="\t",
                    index=False)

    # ------ SPRITES ------
    sprite_img = _make_sprite(images, sprite_size)    
    sprite_img.save(os.path.join(log_dir, "sprites.png"))
    
    # ------ TENSORBOARD CONFIG ------
    # store the data in a tensor
    feature_tensor = tf.Variable(features, name="embeddings")
    # save it as a checkpoint
    checkpoint = tf.train.Checkpoint(embedding=feature_tensor)
    checkpoint.save(os.path.join(log_dir, "embedding.ckpt"))
    # configure projector
    config = projector.ProjectorConfig()
    embedding = config.embeddings.add()
    embedding.tensor_name =  "embedding/.ATTRIBUTES/VARIABLE_VALUE"
    embedding.metadata_path = "metadata.tsv"
    embedding.sprite.image_path = "sprites.png" 
    embedding.sprite.single_image_dim.extend([sprite_size, sprite_size])
    projector.visualize_embeddings(log_dir, config)









