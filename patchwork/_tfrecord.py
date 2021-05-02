# -*- coding: utf-8 -*-
import tensorflow as tf
from patchwork.loaders import _image_file_dataset


def save_dataset_to_tfrecords(imfiles, outdir, num_shards=10, imshape=(256,256), num_channels=3, 
                              norm=255, num_parallel_calls=None):
    """
    Save a dataset to tfrecord files. Wrapper for tf.data.experimental.save().
    
    :imfiles: list of paths to image files, or tensorflow dataset to spool to disk
    :outdir: top-level directory to save tfrecords to
    :num_shards: how many shards to divide the dataset into
    :imshape: (tuple) image dimensions in H,W
    :num_channels: (int) number of image channels
    :norm: (int or float) normalization constant for images (for rescaling to
               unit interval)
    :num_parallel_calls: (int) number of threads for loader mapping
    """
    if not isinstance(imfiles, tf.data.Dataset):
        imfiles = _image_file_dataset(imfiles, imshape=imshape, norm=norm, 
                                      num_channels=num_channels,
                                      num_parallel_calls=num_parallel_calls).prefetch(num_parallel_calls)
        
    def _shardfunc(x):
        return tf.random.uniform((), minval=0, maxval=num_shards, dtype=tf.int64)
    
    tf.data.experimental.save(imfiles, outdir, compression="GZIP", shard_func=_shardfunc)
