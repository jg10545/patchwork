"""

                _augment.py


"""
import tensorflow as tf


DEFAULT_AUGMENT_PARAMS = {
        "max_brightness_delta":0.2, 
        "contrast_min":0.4, 
        "contrast_max":1.4,
        "max_hue_delta":0.1, 
        "max_saturation_delta":0.5,
        "left_right_flip":True, 
        "up_down_flip":True,
        "rot90":True,
        "zoom_scale":0.2,
        "select_prob":0.5
        }


def _random_zoom(im, imshape, scale=0.1):
    # choose random box values
    xmin = tf.random.uniform(shape=[], minval=0, maxval=scale)
    xmax = tf.random.uniform(shape=[], minval=1-scale, maxval=1)
    ymin = tf.random.uniform(shape=[], minval=0, maxval=scale)
    ymax = tf.random.uniform(shape=[], minval=1-scale, maxval=1)
    box = tf.stack([[ymin, xmin, ymax, xmax]])

    resized = tf.image.crop_and_resize(tf.expand_dims(im, 0), 
                                       box, 
                                       [0],
                                       imshape)
    return tf.squeeze(resized, axis=0)
    
def _augment(im, imshape=None, max_brightness_delta=0.2, contrast_min=0.4, contrast_max=1.4,
             max_hue_delta=0.1,  max_saturation_delta=0.5,
             left_right_flip=True, up_down_flip=True, rot90=True, zoom_scale=0.1,
             select_prob=0.5):
    """
    Macro to do random image augmentation
    """
    # built-in methods
    if max_brightness_delta:
        if tf.random.uniform(()) <= select_prob:
            im = tf.image.random_brightness(im, max_brightness_delta)
    if bool(contrast_min) & bool(contrast_max):
        if tf.random.uniform(()) <= select_prob:
            im = tf.image.random_contrast(im, contrast_min, contrast_max)
    if max_hue_delta:
        if tf.random.uniform(()) <= select_prob:
            im = tf.image.random_hue(im, max_hue_delta)
    if max_saturation_delta:
        if tf.random.uniform(()) <= select_prob:
            im = tf.image.random_saturation(im, 1-max_saturation_delta,
                                        1+max_saturation_delta)
    if left_right_flip:
        im = tf.image.random_flip_left_right(im)
    if up_down_flip:
        im = tf.image.random_flip_up_down(im)
    if rot90:
        if tf.random.uniform(()) <= select_prob:
            theta = tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32)
            im = tf.image.rot90(im, theta)
    if zoom_scale > 0:
        if tf.random.uniform(()) <= select_prob:
            im = _random_zoom(im, imshape, zoom_scale)
    
    # custom- random rotation, distortion, crop, and rescale
    #if rotate_degrees:
    #    im = _random_rotate(im)
    #if distort_theta:
    #    im = _random_distort(im, distort_theta)
    #if crop_delta:
    #    im = _random_crop(im, crop_delta)
        
    # some augmentation can put values outside unit interval
    im = tf.minimum(im, 1)
    im = tf.maximum(im, 0)
    return im


def augment_function(imshape, params=True):
    """
    Define an augmentation function from a dictionary of parameters
    """
    if params == True:
        params = DEFAULT_AUGMENT_PARAMS
    #return lambda x: _augment(x, **params)
    @tf.function
    def _aug(x):
        return _augment(x, imshape=imshape, **params)
    return _aug
