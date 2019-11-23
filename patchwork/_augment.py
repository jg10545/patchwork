"""

                _augment.py


"""
import tensorflow as tf


DEFAULT_AUGMENT_PARAMS = {
        "max_brightness_delta":0.2, 
        "contrast_min":0.4, 
        "contrast_max":1.4,
        "max_hue_delta":0.1, 
        "jpeg_quality_delta":10, 
        "max_saturation_delta":0.5,
        "left_right_flip":True, 
        "up_down_flip":True
        }

#def _random_rotate(im, rotate_degrees):
#    theta = 3.14159*rotate_degrees/180
#    theta = tf.random_uniform((), minval=-1*rotate_degrees, maxval=rotate_degrees)
#    return tf.contrib.image.rotate(im, theta)

#def _random_crop(im, crop_delta):
#    imshape = im.get_shape().as_list()
#    im = tf.image.crop_and_resize(tf.expand_dims(im, 0),
#                        [[tf.random.uniform((), minval=0, maxval=crop_delta),
#                          tf.random.uniform((), minval=0, maxval=crop_delta),
#                          tf.random.uniform((), minval=1-crop_delta, maxval=1),
#                          tf.random.uniform((), minval=1-crop_delta, maxval=1)]],
#                        [0],
#                        imshape[:2])
#    return tf.squeeze(im, 0)
    
#def _random_distort(im, distort_theta):
#    theta2 = tf.random.uniform((), minval=-1*distort_theta, maxval=distort_theta)
#    trans = [1, tf.sin(theta2), 0, 0, tf.cos(theta2), 0, 0, 0]
#    return tf.contrib.image.transform(im, transforms=trans)
    
def _augment(im, max_brightness_delta=0.2, contrast_min=0.4, contrast_max=1.4,
             max_hue_delta=0.1, jpeg_quality_delta=10, max_saturation_delta=0.5,
             left_right_flip=True, up_down_flip=True):
    """
    Macro to do random image augmentation
    """
    # built-in methods
    if max_brightness_delta:
        im = tf.image.random_brightness(im, max_brightness_delta)
    if bool(contrast_min) & bool(contrast_max):
        im = tf.image.random_contrast(im, contrast_min, contrast_max)
    if max_hue_delta:
        im = tf.image.random_hue(im, max_hue_delta)
    if jpeg_quality_delta:
        im = tf.image.random_jpeg_quality(im, jpeg_quality_delta, 
                                          100-jpeg_quality_delta)
    if max_saturation_delta:
        im = tf.image.random_saturation(im, 1-max_saturation_delta,
                                        1+max_saturation_delta)
    if left_right_flip:
        im = tf.image.random_flip_left_right(im)
    if up_down_flip:
        im = tf.image.random_flip_up_down(im)
    
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


def augment_function(params=True):
    """
    Define an augmentation function from a dictionary of parameters
    """
    if params == True:
        params = DEFAULT_AUGMENT_PARAMS
    return lambda x: _augment(x, **params)
