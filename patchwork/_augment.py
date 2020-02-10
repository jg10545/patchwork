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
        "zoom_scale":0.3,
        "select_prob":0.5
        }


def _poisson(lam):
    return tf.random.poisson(shape=[], lam=lam, 
                               dtype=tf.int32)

def _random_zoom(im, imshape, scale=0.1):
    """
    Randomly pad and then randomly crop an image
    """
    im = tf.expand_dims(im, 0)
    
    # first, pad the image
    toppad = _poisson(imshape[0]*scale/2)
    leftpad = _poisson(imshape[1]*scale/2)
    bottompad = _poisson(imshape[0]*scale/2)
    rightpad = _poisson(imshape[1]*scale/2)
    padded = tf.image.pad_to_bounding_box(
        im,
        toppad,
        leftpad,
        toppad+imshape[0]+bottompad,
        leftpad+imshape[1]+rightpad
    )
    
    # second, crop the image
    xmin = tf.random.uniform(shape=[], minval=0, maxval=scale)
    xmax = tf.random.uniform(shape=[], minval=1-scale, maxval=1)
    ymin = tf.random.uniform(shape=[], minval=0, maxval=scale)
    ymax = tf.random.uniform(shape=[], minval=1-scale, maxval=1)
    box = tf.stack([[ymin, xmin, ymax, xmax]])

    resized = tf.image.crop_and_resize(padded, 
                                       box, 
                                       [0],
                                       imshape)
    return tf.squeeze(resized, axis=0)

    
def _choose(prob):
    return tf.random.uniform(()) <= prob

def _augment(im, imshape=None, max_brightness_delta=False, contrast_min=False, contrast_max=False,
             max_hue_delta=False,  max_saturation_delta=False,
             left_right_flip=False, up_down_flip=False, rot90=False, zoom_scale=False,
             select_prob=1.):
    """
    Macro to do random image augmentation
    """
    # built-in methods
    if max_brightness_delta:
        if _choose(select_prob):
            im = tf.image.random_brightness(im, max_brightness_delta)
    if bool(contrast_min) & bool(contrast_max):
        if _choose(select_prob):
            im = tf.image.random_contrast(im, contrast_min, contrast_max)
    if max_hue_delta:
        if _choose(select_prob):
            im = tf.image.random_hue(im, max_hue_delta)
    if max_saturation_delta:
        if _choose(select_prob):
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
        if _choose(select_prob):
            im = _random_zoom(im, imshape, zoom_scale)
        
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
    @tf.function
    def _aug(x):
        return _augment(x, imshape=imshape, **params)
    return _aug

