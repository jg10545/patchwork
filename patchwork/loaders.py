"""

                _loaders.py

Code for loading data into tensorflow datasets
"""
import numpy as np
import tensorflow as tf
#from PIL import Image
from patchwork._util import tiff_to_array
from patchwork._augment import augment_function
from patchwork._tasks import _rotate

import os


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


def _build_load_function(path, imshape, norm, num_channels, single_channel,
                         augment=False):
    """
    macro to build a function that handles all the loading. the
    load function takes in two inputs (file path and integer specifying file
    type) and returns the loaded image
    
    Note- this used to return a tf.function, but when used within
    a tensorflow Dataset I'd occasionally get "InvalidArgumentError: 
        The graph couldn't be sorted in topological order. 
        [Op:OptimizeDataset]" errors
    """
    if path.lower().endswith(".jpg"):
        decode = tf.io.decode_jpeg
    elif path.lower().endswith(".png"):
        decode = tf.io.decode_png
    elif path.lower().endswith(".gif"):
        decode = tf.io.decode_gif
    else:
        assert False, "unsupported file type"
    
    def _load_img(x,y):
        loaded = tf.io.read_file(x)
        img = decode(loaded)
        if single_channel:
            #resized = tf.concat(num_channels*[resized], -1)
            img = tf.concat(num_channels*[img], -1)
        # normalize
        img = tf.cast(img[:,:,:num_channels], tf.float32)/norm
        #img = tf.reshape(normed, (imshape[0], imshape[1], num_channels))
        if augment:
            img = augment_function(imshape, augment)(img)
        # TWO RESIZING OPERATIONS HERE
        # first: tf.image.resize(), which will resample the image if it
        # doesn't have the right dimensions
        img = tf.image.resize(img, imshape)
        # in some cases tf.image.resize() will return tensors of shape
        # [imshape[0], imshape[1], None], so now we'll specify the
        # channel dimension explicitly.
        img = tf.reshape(img, [imshape[0], imshape[1], num_channels])
        return img,y
    
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
    # PRE-BUILT DATASET CASE
    if isinstance(fps, tf.data.Dataset):
        ds = fps
        if augment:
            _aug = augment_function(imshape, augment)
            ds = ds.map(_aug, num_parallel_calls=num_parallel_calls)
    # DIRECTORY OF TFRECORD FILES CASE
    elif isinstance(fps, str):
        if augment:
            augment = augment_function(imshape, augment)
        ds = load_dataset_from_tfrecords(fps, imshape, num_channels,
                                         num_parallel_calls=num_parallel_calls,
                                         map_fn=augment)
    # LIST OF FILES CASE: list of filepaths (probably what almost 
    # always will get used)
    elif isinstance(fps[0], str):
        if ys is None:
            no_labels = True
            ys = np.zeros(len(fps), dtype=np.int64)
        else:
            no_labels = False
        # get an integer index for each filepath
        #imtypes = _generate_imtypes(fps)
        ds = tf.data.Dataset.from_tensor_slices((fps, ys))
        # do the shuffling before loading so we can have a big queue without
        # taking up much memory
        if shuffle:
            ds = ds.shuffle(len(fps))
        _load_img = _build_load_function(fps[0], imshape, norm, num_channels, 
                                         single_channel, augment)
        #ds = ds.map(lambda x,t,y: (_load_img(x),y), 
        #            num_parallel_calls=num_parallel_calls)
        ds = ds.map(_load_img, num_parallel_calls=num_parallel_calls)
        # if no labels were passed, strip out the y.
        if no_labels:
            ds = ds.map(lambda x,y: x)
            
    else:
        assert False, "what are these inputs"
            
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
    # CUSTOM DATASET CASE 
    if isinstance(fps, tf.data.Dataset):
        num_steps = None
    # SINGLE-INPUT CASE (DEFAULT)
    elif isinstance(fps[0], str):
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



def _get_features(fcn, downstream_labels, avpool=False, 
                  return_images=False, **input_config):
    """
    Load features into memory either from a dictionary mapping filepaths
    to labels, or a dataset that generates batches of (image, label) pairs
    
    :fcn: Keras fully-convolutional network
    :downstream_labels: dictionary mapping image file paths to labels, or a 
        dataset returning image/label pairs
    :avpool: average-pool feature tensors before fitting linear model. if False, flatten instead.
    :return_images: if True, also return a big array of all the stacked-up 
        raw images
    :input_config: kwargs for patchwork.loaders.dataset()
    
    Returns
    :acc: float; test accuracy
    :cm: 2D numpy array; confusion matrix
    """
    # if input is a dictionary, build the dataset
    if not isinstance(downstream_labels, tf.data.Dataset):
        X = list(downstream_labels.keys())
        Y = list(downstream_labels.values())
        # multiple input case
        if "," in X[0]:
            X0 = [x.split(",")[0] for x in X]
            X1 = [x.split(",")[1] for x in X]
            ds, num_steps = dataset([X0, X1], ys=Y,
                                shuffle=False,
                                **input_config)
        # single input case
        else:
            ds, num_steps = dataset(X, ys=Y, shuffle=False,
                                            **input_config)
    else:
        ds = downstream_labels
    # run labeled images through the network and flatten or average
    features = []
    labels = []
    images = []
    for x,y in ds:
        features.append(fcn(x).numpy())
        labels.append(y.numpy())
        if return_images: images.append(x)
    features = np.concatenate(features, 0)
    labels = np.concatenate(labels, 0)
    labels = np.array([str(l) for l in labels.ravel()])
    if return_images: images = np.concatenate(images, 0)
    
    if avpool:
        features = features.mean(axis=1).mean(axis=1)
    else:
        features = features.reshape(features.shape[0], -1)   
        
    if return_images:
        return features, labels, images
    else:
        return features, labels
    


def _build_rotation_dataset(imfiles, imshape=(256,256), batch_size=256, 
                      num_parallel_calls=None, norm=255,
                      num_channels=3, augment=False,
                      single_channel=False):
    """
    """
    ds = _image_file_dataset(imfiles, shuffle=True, augment=augment, 
                             imshape=imshape, norm=norm,
                             single_channel=single_channel,
                             num_parallel_calls=num_parallel_calls)
        
    ds = ds.map(_rotate, num_parallel_calls=num_parallel_calls)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(1)
    return ds
    

def _get_rotation_features(fcn, imfiles, avpool=True, **input_config):
    """
    Generate features without labels, using a self-supervised rotation task. See "Evaluating 
    Self-Supervised Pretraining Without Using Labels" by Reed et al for details.
    
    Currently only supports single-input networks.
    
    :fcn: Keras fully-convolutional network
    :imfiles: list of image filepaths, or a dataset that generates images
    :avpool: average-pool feature tensors before fitting linear model. if False, flatten instead.
    :input_config: kwargs for patchwork.loaders.dataset()
    
    Returns
    :features: 2D array of flattened or averaged feature vectors
    :labels: 1D array of labels
    """
    inpt = {k:input_config[k] for k in input_config if k != "batch_size"}
    # if input is a dictionary, build the dataset
    if not isinstance(imfiles, tf.data.Dataset):
        ds = _image_file_dataset(imfiles, shuffle=False, augment=False, **inpt)
    else:
        ds = imfiles
        
    ds = ds.map(_rotate, num_parallel_calls=input_config.get("num_parallel_calls", None))
    ds = ds.batch(input_config.get("batch_size", 32))
    
    # run labeled images through the network and flatten or average
    features = []
    labels = []
    for x,y in ds:
        features.append(fcn(x).numpy())
        labels.append(y.numpy())
    features = np.concatenate(features, 0)
    labels = np.concatenate(labels, 0).ravel()

    if avpool:
        features = features.mean(axis=1).mean(axis=1)
    else:
        features = features.reshape(features.shape[0], -1)   
        
    return features, labels



def _fixmatch_unlab_dataset(fps, weak_aug, str_aug, imshape=(256,256),
                            num_parallel_calls=None, norm=255, 
                            num_channels=3, single_channel=False,
                            batch_size=64):
    """
    Macro to build the weak/strong augmented pairs for FixMatch
    semisupervised learning
    """
    _weakaug = augment_function(imshape, weak_aug)
    _strongaug = augment_function(imshape, str_aug)
    def aug_pair(x):
        return _weakaug(x), _strongaug(x)
    
    
    ds = _image_file_dataset(fps, imshape=imshape, 
                             num_parallel_calls=num_parallel_calls, 
                             norm=norm, num_channels=num_channels,
                             shuffle=True, 
                             single_channel=single_channel, 
                             augment=False)
    ds = ds.map(aug_pair, num_parallel_calls=num_parallel_calls)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(1)
    return ds

    
    
def load_dataset_from_tfrecords(record_dir, imshape, num_channels,
                                shuffle=2048, num_parallel_calls=None,
                                map_fn=None, num_images=1):
    """
    Load a directory structure of tfrecord files (like you'd build with save_to_tfrecords) 
    into a tensorflow dataset. Wrapper for tf.data.TFRecordDataset().
    
    Assumes tfrecord files are saved with .tfrecord or .snapshot.
    
    :record_dir: top-level directory containing record files
    :imshape: (H,W) dimensions of image
    :num_channels: number of channels
    :shuffle: if not False, size of shuffle queue
    :num_parallel_calls: number of parallel readers/mappers for loading and parsing
        the dataset
    :map_fn: function to map across dataset during loading (for example, for augmentation)
    """
    # single-image case: dataset returns a single tensor
    if num_images == 1:
        element_spec = tf.TensorSpec(shape=(imshape[0],imshape[1],num_channels),
                                     dtype=tf.float32)
    # multiple-image case: dataset returns a TUPLE of `num_images` tensors
    else:
        element_spec = tuple(tf.TensorSpec(shape=(imshape[0],imshape[1],num_channels),
                                     dtype=tf.float32) for 
                         _ in range(num_images))
    # note that this function may change in the future
    ds = tf.data.experimental.load(record_dir, element_spec, compression="GZIP")
    # if a map function was included, map across the dataset
    if map_fn is not None:
        ds = ds.map(map_fn, num_parallel_calls=num_parallel_calls)
    if shuffle:
        ds = ds.shuffle(shuffle)
    return ds