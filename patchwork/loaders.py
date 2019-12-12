"""

                _loaders.py

Code for loading data into tensorflow datasets
"""
import numpy as np
import tensorflow as tf
from PIL import Image
from patchwork._util import tiff_to_array

from patchwork._augment import augment_function



def _generate_imtypes(fps):
    """
    Input a list of filepaths and return an array mapping
    filetypes to an integer index:
        
        png 0
        jpg 1
        gif 2
        tif 3
    """
    imtypes = np.zeros(len(fps), dtype=np.int64)
    for i in range(len(fps)):
        t = fps[i].lower()
        if ".jpg" in t:
            imtypes[i] = 1
        elif ".gif" in t:
            imtypes[i] = 2
        elif ".tif" in t:
            imtypes[i] = 3
    return imtypes

def _image_file_dataset(fps, imshape=(256,256), 
                 num_parallel_calls=None, norm=255,
                 num_channels=3, shuffle=False):
    """
    Basic tool to load images into a tf.data.Dataset using
    PIL.Image or gdal instead of the tensorflow decode functions
    
    :fps: list of filepaths
    :imshape: constant shape to resize images to
    :num_parallel_calls: number of processes to use for loading
    :norm: value for normalizing images
    :num_channels: channel depth to truncate images to
    :shuffle: whether to shuffle the dataset
    
    Returns images as a 3D float32 tensor
    """
    # get an integer index for each filepath
    imtypes = _generate_imtypes(fps)
    ds = tf.data.Dataset.from_tensor_slices((fps, imtypes))
    # do the shuffling before loading so we can have a big queue without
    # taking up much memory
    if shuffle:
        ds = ds.shuffle(len(fps))
    
    # helper function for resizing images
    def _resize(img):
        return tf.image.resize(img, imshape)
    # helper function for loading tiffs
    def _load_tif(f):
        return _resize(tiff_to_array(f.numpy().decode("utf-8") , swapaxes=True, 
                                 norm=norm, num_channels=num_channels))
    load_tif = lambda x: tf.py_function(_load_tif, [x], tf.float32)
    
    # main loading map function
    @tf.function
    def _load_img(x, t):
        #print("tracing")
        loaded = tf.io.read_file(x)
        # jpg
        if t == 1:
            decoded = tf.io.decode_jpeg(loaded)
            resized = _resize(decoded)
        # gif
        elif t == 2:
            decoded = tf.io.decode_gif(loaded)
            resized = _resize(decoded)
        # tif
        elif t == 3:
            resized = load_tif(x)
        # png
        else:
            decoded = tf.io.decode_png(loaded)
            resized = _resize(decoded)
        return tf.cast(resized, tf.float32)/255

    ds = ds.map(lambda x,y: _load_img(x,y), num_parallel_calls=num_parallel_calls)
    #tensorshape = [imshape[0], imshape[1], num_channels]
    #ds = ds.map(lambda x: tf.reshape(x, tensorshape), num_parallel_calls=num_parallel_calls)
    return ds




def dataset(fps, ys = None, imshape=(256,256), num_channels=3, 
                 num_parallel_calls=None, norm=255, batch_size=256,
                 augment=False, unlab_fps=None, shuffle=False):
    """
    return a tf dataset that iterates over a list of images once
    
    :fps: list of filepaths
    :ys: array of corresponding labels
    :imshape: constant shape to resize images to
    :num_channels: channel depth of images
    :batch_size: just what you think it is
    :augment: augmentation parameters (or False to disable)
    :unlab_fps: list of filepaths (same length as fps) for semi-
        supervised learning
    :shuffle: whether to shuffle the dataset
    
    Returns
    :ds: tf.data.Dataset object to iterate over data. The dataset returns
        (x,y) tuples unless unlab_fps is included, in which case the structure
        will be ((x, x_unlab), (y,y))
    :num_steps: number of steps (for passing to tf.keras.Model.fit())
    """
    if augment:
        _aug = augment_function(imshape, augment)
    ds = _image_file_dataset(fps, imshape=imshape, num_channels=num_channels, 
                      num_parallel_calls=num_parallel_calls, norm=norm,
                      shuffle=shuffle)
    
    if augment: ds = ds.map(_aug, num_parallel_calls=num_parallel_calls)
        
    if unlab_fps is not None:
        u_ds = _image_file_dataset(unlab_fps, imshape=imshape, num_channels=num_channels, 
                      num_parallel_calls=num_parallel_calls, norm=norm)
        if augment: u_ds = u_ds.map(_aug, num_parallel_calls=num_parallel_calls)
        ds = tf.data.Dataset.zip((ds, u_ds))
        
    if ys is not None:
        ys = tf.data.Dataset.from_tensor_slices(ys)
        if unlab_fps is not None:
            ys = ds.zip((ys,ys))
            #ys = ds.zip((u_ds,ys))
        ds = ds.zip((ds, ys))
        
    ds = ds.batch(batch_size)
    ds = ds.prefetch(1)
    
    num_steps = int(np.ceil(len(fps)/batch_size))
    return ds, num_steps






def stratified_training_dataset(fps, y, imshape=(256,256), num_channels=3, 
                 num_parallel_calls=None, batch_size=256, mult=10,
                    augment=True, norm=255):
    """
    Training dataset for DeepCluster.
    Build a dataset that provides stratified samples over labels
    
    :fps: list of strings containing paths to image files
    :y: array of cluster assignments- should have same length as fp
    :imshape: constant shape to resize images to
    :num_channels: channel depth of images
    :batch_size: just what you think it is
    :mult: not in paper; multiplication factor to increase
        number of steps/epoch. set to 1 to get paper algorithm
    :augment:
        
    Returns
    :ds: tf.data.Dataset object to iterate over data
    :num_steps: number of steps (for passing to tf.keras.Model.fit())
    """
    # sample indices to use
    indices = np.arange(len(fps))
    K = y.max()+1
    samples_per_cluster = mult*int(len(fps)/K)
    
    sampled_indices = []
    sampled_labels = []
    # for each cluster
    for k in range(K):
        # find indices of samples assigned to it
        cluster_inds = indices[y == k]
        # only sample if at least one is assigned. note that
        # the deepcluster paper takes an extra step here.
        if len(cluster_inds) > 0:
            samps = np.random.choice(cluster_inds, size=samples_per_cluster,
                            replace=True)
            sampled_indices.append(samps)
            sampled_labels.append(k*np.ones(len(samps), dtype=np.int64))
    # concatenate sampled indices for each cluster
    sampled_indices = np.concatenate(sampled_indices, 0)    
    sampled_labels = np.concatenate(sampled_labels, 0)
    # and shuffle their order together
    reorder = np.random.choice(np.arange(len(sampled_indices)),
                          size=len(sampled_indices), replace=False)
    sampled_indices = sampled_indices[reorder]
    sampled_labels = sampled_labels[reorder]
    fps = np.array(fps)[sampled_indices]
    
    # NOW CREATE THE DATASET
    im_ds = _image_file_dataset(fps, imshape=imshape, num_channels=num_channels, 
                      num_parallel_calls=num_parallel_calls, norm=norm, 
                      shuffle=False)

    if augment:
        #im_ds = im_ds.map(_augment, num_parallel_calls)
        _aug = augment_function(imshape, augment)
        im_ds = im_ds.map(_aug, num_parallel_calls=num_parallel_calls)
    lab_ds = tf.data.Dataset.from_tensor_slices(sampled_labels)
    ds = tf.data.Dataset.zip((im_ds, lab_ds))
    ds = ds.batch(batch_size)
    ds = ds.prefetch(1)
    
    num_steps = int(np.ceil(len(sampled_indices)/batch_size))
    return ds, num_steps