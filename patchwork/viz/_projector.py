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
from patchwork.loaders import _get_features



def _make_sprite(images, spritesize=50):
    """
    Input a 4D tensor, output a sprite image. 
    assumes your pictures are all the
    same size and square
    """
    num_sprites = images.shape[0]
    imsize = images.shape[1]
    gridsize = int(np.ceil(np.sqrt(num_sprites)))
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
    Output embeddings in a format the tensorboard projector can use.
    
    :fcn: keras Model; feature extractor to generate embeddings
    :labeldict: dictionary mapping filepaths to labels, or a dataset that
        generates batches of (image, label) pairs
    :log_dir: path to directory to save in
    :proj_dim: if above zero, use PCA to project embeddings down to this
        dimension before saving
    :sprite_size: pixel size for saving sprites
    """
    # ------ LOAD FEATURES AND LABELS INTO MEMORY -----
    features, labels, images = _get_features(fcn, labeldict, return_images=True,
                                             **input_config)

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
    labels = list(labeldict.values())
    filepaths = list(labeldict.keys())
    metadata = pd.DataFrame({"label":labels,
                             "traintest":traintest})
    # if possible, save out parent and grandparent directories
    if len(filepaths[0].split("/")) > 1:
        metadata["parent"] = [f.split("/")[-2] for f in filepaths]
    if len(filepaths[0].split("/")) > 2:
        metadata["grandparent"] = [f.split("/")[-3] for f in filepaths]
    
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









