# -*- coding: utf-8 -*-
import tensorflow as tf

from patchwork.loaders import _image_file_dataset


def _rotate(x, foo=False, **kwargs):
    # create a labeled rotated image
    theta = tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32)
    return tf.image.rot90(x, theta), theta

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
    
