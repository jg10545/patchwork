"""

                _loaders.py

Code for loading data into tensorflow datasets
"""
import numpy as np
import tensorflow as tf
#from PIL import Image
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
        if (".jpg" in t) or ("jpeg" in t):
            imtypes[i] = 1
        elif ".gif" in t:
            imtypes[i] = 2
        elif ".tif" in t:
            imtypes[i] = 3
    return imtypes


def _build_load_function(imshape, norm, num_channels, single_channel,
                         augment=False):
    """
    macro to build a tf.function that handles all the loading. the
    load function takes in two inputs (file path and integer specifying file
    type) and returns the loaded image
    """
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
            
        if single_channel:
            resized = tf.concat(num_channels*[resized], -1)
        normed = tf.cast(resized[:,:,:num_channels], tf.float32)/norm
        img = tf.reshape(normed, (imshape[0], imshape[1], num_channels))
        if augment:
            img = augment_function(imshape, augment)(img)
        
        return img
    
    return _load_img


def _select(x,i):
    # convenience function to simplify code in the dual-input case.
    # if the config is a float or int, use it for both inputs
    if isinstance(x,float)|isinstance(x,int):
        return x
    # if the config is a container- return the ith element
    else:
        return x[i]


def _image_file_dataset(fps, ys=None, imshape=(256,256), 
                 num_parallel_calls=None, norm=255,
                 num_channels=3, shuffle=False,
                 single_channel=False, augment=False):
    """
    Basic tool to load images into a tf.data.Dataset using
    PIL.Image or gdal instead of the tensorflow decode functions
    
    :fps: list of filepaths
    :ys: optional list of labels
    :imshape: constant shape to resize images to
    :num_parallel_calls: number of processes to use for loading
    :norm: value for normalizing images
    :num_channels: channel depth to truncate images to
    :shuffle: whether to shuffle the dataset
    :single_channel: if True, expect a single-channel input image and 
        stack it num_channels times.
    :augment: optional, dictionary of augmentation parameters
    
    Returns tf.data.Dataset object with structure (x,y) if labels were passed, 
        and (x) otherwise. images (x) are a 3D float32 tensor and labels
        should be a 0D int64 tensor
    """    
    # SINGLE-INPUT CASE (DEFAULT)
    if isinstance(fps[0], str):
        if ys is None:
            no_labels = True
            ys = np.zeros(len(fps), dtype=np.int64)
        else:
            no_labels = False
        # get an integer index for each filepath
        imtypes = _generate_imtypes(fps)
        ds = tf.data.Dataset.from_tensor_slices((fps, imtypes, ys))
        # do the shuffling before loading so we can have a big queue without
        # taking up much memory
        if shuffle:
            ds = ds.shuffle(len(fps))
        _load_img = _build_load_function(imshape, norm, num_channels, 
                                         single_channel, augment)
        ds = ds.map(lambda x,t,y: (_load_img(x,t),y), 
                    num_parallel_calls=num_parallel_calls)
        # if no labels were passed, strip out the y.
        if no_labels:
            ds = ds.map(lambda x,y: x)
    # DUAL-INPUT CASE
    else:
        # let's do some input checking here
        assert len(fps) == 2, "only single or double input currently supported"
        if isinstance(imshape[0], int): imshape = (imshape, imshape)
        
        if ys is None:
            no_labels = True
            ys = np.zeros(len(fps[0]), dtype=np.int64)
        else:
            no_labels = False
        # parse out filetypes
        imtypes0 = _generate_imtypes(fps[0])
        imtypes1 = _generate_imtypes(fps[1])
        ds = tf.data.Dataset.from_tensor_slices((fps[0], imtypes0, 
                                                 fps[1], imtypes1,
                                                 ys))
        if shuffle:
            ds = ds.shuffle(len(fps))
        _load_img0 = _build_load_function(imshape[0], _select(norm,0), 
                                          _select(num_channels,0), 
                                          _select(single_channel,0),
                                          augment)
        _load_img1 = _build_load_function(imshape[1], _select(norm,1), 
                                          _select(num_channels,1), 
                                          _select(single_channel,1),
                                          augment)
                                          
        ds = ds.map(lambda x0,t0,x1,t1,y: (
                    _load_img0(x0,t0), _load_img1(x1,t1),y), 
                    num_parallel_calls=num_parallel_calls)
        # if no labels were passed, strip out the y.
        if no_labels:
            ds = ds.map(lambda x0,x1,y: (x0,x1))
    
    return ds




def dataset(fps, ys = None, imshape=(256,256), num_channels=3, 
                 num_parallel_calls=None, norm=255, batch_size=256,
                 augment=False, shuffle=False,
                 single_channel=False):
    """
    return a tf dataset that iterates over a list of images once
    
    :fps: list of filepaths
    :ys: array of corresponding labels
    :imshape: constant shape to resize images to
    :num_channels: channel depth of images
    :batch_size: just what you think it is
    :augment: augmentation parameters (or True for defaults, or False to disable)
    :shuffle: whether to shuffle the dataset
    :single_channel: if True, expect a single-channel input image and 
        stack it num_channels times.
    
    Returns
    :ds: tf.data.Dataset object to iterate over data. The dataset returns
        (x,y) tuples unless unlab_fps is included, in which case the structure
        will be ((x, x_unlab), y)
    :num_steps: number of steps (for passing to tf.keras.Model.fit())
    """
    ds = _image_file_dataset(fps, ys=ys, imshape=imshape, num_channels=num_channels, 
                      num_parallel_calls=num_parallel_calls, norm=norm,
                      shuffle=shuffle, single_channel=single_channel,
                      augment=augment)
    # SINGLE-INPUT CASE (DEFAULT)
    if isinstance(fps[0], str):
        num_steps = int(np.ceil(len(fps)/batch_size))

    # DUAL-INPUT CASE
    else:
        num_steps = int(np.ceil(len(fps[0])/batch_size))
        
    ds = ds.batch(batch_size)
    ds = ds.prefetch(1)
    
    
    return ds, num_steps






def stratified_training_dataset(fps, y, imshape=(256,256), num_channels=3, 
                 num_parallel_calls=None, batch_size=256, mult=10,
                    augment=True, norm=255, single_channel=False):
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
    :augment: augmentation parameters (or True for defaults, or False to disable)
    :single_channel: if True, expect a single-channel input image and 
        stack it num_channels times.
        
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
    # SINGLE-INPUT CASE
    if isinstance(fps[0], str):
        fps = np.array(fps)[sampled_indices]
    # DUAL-INPUT CASE
    else:
        fps = [np.array(fps[0])[sampled_indices],
                np.array(fps[1])[sampled_indices]]
    
    # NOW CREATE THE DATASET
    ds, _ = dataset(fps, ys=sampled_labels,
                      imshape=imshape, num_channels=num_channels, 
                      num_parallel_calls=num_parallel_calls, norm=norm, 
                      shuffle=False, single_channel=single_channel,
                      augment=augment, batch_size=batch_size)

    return ds