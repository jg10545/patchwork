# -*- coding: utf-8 -*-
import tensorflow as tf


def _rotate(x, foo=False, **kwargs):
    # create a labeled rotated image
    theta = tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32)
    return tf.image.rot90(x, theta), theta


