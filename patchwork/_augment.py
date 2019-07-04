"""

                _augment.py


"""
import tensorflow as tf


def _random_rotate(im):
    theta = tf.random_uniform((), minval=-20, maxval=20)
    return tf.contrib.image.rotate(im, theta)

def _random_crop(im):
    imshape = im.get_shape().as_list()
    im = tf.image.crop_and_resize(tf.expand_dims(im, 0),
                        [[tf.random_uniform((), minval=0, maxval=0.15),
                          tf.random_uniform((), minval=0, maxval=0.15),
                          tf.random_uniform((), minval=0.85, maxval=1),
                          tf.random_uniform((), minval=0.85, maxval=1)]],
                        [0],
                        imshape[:2])
    return tf.squeeze(im, 0)
    
def _random_distort(im):
    theta2 = tf.random_uniform((), minval=-0.1, maxval=0.1)
    trans = [1, tf.sin(theta2), 0, 0, tf.cos(theta2), 0, 0, 0]
    return tf.contrib.image.transform(im, transforms=trans)
    
def _augment(im):
    """
    Macro to do random image augmentation
    """
    # built-in methods
    im = tf.image.random_brightness(im, 0.2)
    im = tf.image.random_contrast(im, 0.4, 1.4)
    im = tf.image.random_flip_left_right(im)
    im = tf.image.random_flip_up_down(im)
    
    # custom- random rotation, distortion, crop, and rescale
    im = _random_rotate(im)
    im = _random_distort(im)
    im = _random_crop(im)
        
    # some augmentation can put values outside unit interval
    im = tf.minimum(im, 1)
    im = tf.maximum(im, 0)
    return im

