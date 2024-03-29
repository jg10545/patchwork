# -*- coding: utf-8 -*-
import tensorflow as tf
import warnings
from patchwork.loaders import _image_file_dataset


def save_dataset_to_tfrecords(imfiles, outdir, num_shards=10, imshape=(256,256), num_channels=3, 
                              norm=255, num_parallel_calls=None, gzip=True):
    """
    Save a dataset to tfrecord files. Wrapper for tf.data.experimental.save().
    
    NOTE: THIS DOES NOT HAVE ANY GUARDRAILS! If you pass a dataset, it will write
    until that dataset stops returning data.
    
    :imfiles: list of paths to image files, or tensorflow dataset to spool to disk
    :outdir: top-level directory to save tfrecords to
    :num_shards: how many shards to divide the dataset into
    :imshape: (tuple) image dimensions in H,W
    :num_channels: (int) number of image channels
    :norm: (int or float) normalization constant for images (for rescaling to
               unit interval)
    :num_parallel_calls: (int) number of threads for loader mapping
    :gzip: (bool) whether to compress the tfrecord using GZIP
    """
    if gzip:
        comp = "GZIP"
    else:
        comp = "NONE"
    if not isinstance(imfiles, tf.data.Dataset):
        imfiles = _image_file_dataset(imfiles, imshape=imshape, norm=norm, 
                                      num_channels=num_channels, shuffle=True,
                                      num_parallel_calls=num_parallel_calls)
    else:
        warnings.warn("Make sure your dataset does not return images forever, otherwise this function will definitely murder your hard drive.")
        
    def _shardfunc(*x):
        return tf.random.uniform((), minval=0, maxval=num_shards, dtype=tf.int64)
    
    tf.data.experimental.save(imfiles, outdir, compression=comp, shard_func=_shardfunc)
