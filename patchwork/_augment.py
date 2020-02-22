"""

                _augment.py


"""
import numpy as np
import tensorflow as tf


_DEFAULT_AUGMENT_PARAMS = {
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

DEFAULT_AUGMENT_PARAMS = {
    "gaussian_blur":0.2,
    "drop_color":0.2,
    "gaussian_noise":0.2,
    "sobel_prob":0.1,
    "brightness_delta":0.2,
    "contrast_delta":0.1,
    "saturation_delta":0.1,
    "hue_delta":0.05,
    "flip_left_right":True,
    "flip_up_down":True,
    "rot90":True,
    "zoom_scale":0.2,
    "mask":0.2
}


def _poisson(lam):
    return tf.random.poisson(shape=[], lam=lam, 
                               dtype=tf.int32)


def _choose(prob):
    return tf.random.uniform(()) <= prob


def _drop_color(x, prob=0.25, **kwargs):
    if _choose(prob):
        num_channels = x.shape[-1]
        x_mean = tf.reduce_mean(x, -1)
        x = tf.stack(num_channels*[x_mean], -1)
    return x

def _add_gaussian_noise(x, prob=0.25, **kwargs):
    if _choose(prob):
        x = x + tf.random.normal(x.shape, mean=0., stddev=0.1)
    return x

def _sobelize(x, prob=0.1, **kwargs):
    """
    Augmentation variant of sobelizer function
    
    The first two channels are the sobel filter and
    the third will be zeros (so that it's compatible with
    standard network structures)
    """
    if _choose(prob):
        num_channels = x.shape[-1]
        # run sobel_edges- output will be [1,H,W,C,2]
        x = tf.image.sobel_edges(tf.expand_dims(x, 0))
        # get rid of extra info
        x = tf.squeeze(x,0)
        x = tf.reduce_mean(x, -1)
        x = tf.reduce_mean(x, -1)

        # rescale for contrast
        maxval = tf.reduce_max(x)
        minval = tf.reduce_min(x)
        x = (x - minval)/(maxval-minval)
        # stack back to original number of channels
        x = tf.stack(num_channels*[x], -1)
    return x

def _random_zoom(x, scale=0.1, imshape=(256,256), **kwargs):
    """
    Randomly pad and then randomly crop an image
    """
    x = tf.expand_dims(x, 0)
    
    # first, pad the image
    toppad = _poisson(imshape[0]*scale/2)
    leftpad = _poisson(imshape[1]*scale/2)
    bottompad = _poisson(imshape[0]*scale/2)
    rightpad = _poisson(imshape[1]*scale/2)
    padded = tf.image.pad_to_bounding_box(
        x,
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

def _random_mask(x, prob=0.25,  **kwargs):
    """
    Generates random rectangular masks
    
    """
    if _choose(prob):
        H,W,C = x.shape
    
        dh = tf.random.uniform([], minval=int(H/4), maxval=int(H/2), dtype=tf.int32)
        dw = tf.random.uniform([], minval=int(H/4), maxval=int(W/2), dtype=tf.int32)
    
        xmin = tf.random.uniform([], minval=0, maxval=W-dw, dtype=tf.int32)
        ymin = tf.random.uniform([], minval=0, maxval=H-dh, dtype=tf.int32)
        xmax = xmin + dw
        ymax = ymin + dh
    
        above_xmin = tf.cast(tf.range(0, W) >= xmin, tf.float32)
        above_ymin = tf.cast(tf.range(0, H) >= ymin, tf.float32)

        below_xmax = tf.cast(tf.range(0, W) < xmax, tf.float32)
        below_ymax = tf.cast(tf.range(0, H) < ymax, tf.float32)
    
        prod1 = tf.matmul(tf.expand_dims(above_ymin,-1),tf.expand_dims(above_xmin,0))
        prod2 = tf.matmul(tf.expand_dims(below_ymax,-1),tf.expand_dims(below_xmax,0))
        mask = tf.expand_dims(prod1*prod2,-1)
    
        x =  x*(1-mask) + 0.5*mask
    return x

def _gaussian_blur(x, prob=0.25, **kwargs):
    if _choose(prob):
        kernel = np.array([[0.00000067, 0.00002292, 0.00019117, 0.00038771, 0.00019117, 0.00002292, 0.00000067],
                       [0.00002292, 0.00078633, 0.00655965, 0.01330373, 0.00655965, 0.00078633, 0.00002292],
                       [0.00019117, 0.00655965, 0.05472157, 0.11098164, 0.05472157, 0.00655965, 0.00019117],
                       [0.00038771, 0.01330373, 0.11098164, 0.22508352, 0.11098164, 0.01330373, 0.00038771],
                       [0.00019117, 0.00655965, 0.05472157, 0.11098164, 0.05472157, 0.00655965, 0.00019117],
                       [0.00002292, 0.00078633, 0.00655965, 0.01330373, 0.00655965, 0.00078633, 0.00002292],
                       [0.00000067, 0.00002292, 0.00019117, 0.00038771, 0.00019117, 0.00002292, 0.00000067]])
        zeros = np.zeros((7,7))
        kernel = np.sqrt(kernel)
        kernel /=kernel.sum()
        kernel = np.stack([
            np.stack([kernel, zeros, zeros], -1),
            np.stack([zeros, kernel, zeros], -1),
            np.stack([zeros, zeros, kernel], -1)
        ], -1)
        conv = tf.nn.conv2d(tf.expand_dims(x,0), kernel, strides=[1, 1, 1, 1], padding="SAME")
        x = tf.squeeze(conv, 0)
    return x

def _random_brightness(x, brightness_delta=0.2, **kwargs):
    return tf.image.random_brightness(x, brightness_delta)

def _random_contrast(x, contrast_delta=0.4, **kwargs):
    return tf.image.random_contrast(x, 1-contrast_delta, 1+contrast_delta)

def _random_saturation(x, saturation_delta=0.5, **kwargs):
    return tf.image.random_saturation(x, 1-saturation_delta,
                                        1+saturation_delta)

def _random_hue(x, hue_delta=0.1, **kwargs):
    return tf.image.random_hue(x, hue_delta)

def _random_left_right_flip(x, foo=False, **kwargs):
    return tf.image.random_flip_left_right(x)

def _random_up_down_flip(x, foo=False, **kwargs):
    return tf.image.random_flip_up_down(x)

def _random_rotate(x, foo=False, **kwargs):
    theta = tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32)
    return tf.image.rot90(x, theta)

SINGLE_AUG_FUNC = {
    "gaussian_blur":_gaussian_blur,
    "drop_color":_drop_color,
    "gaussian_noise":_add_gaussian_noise,
    "sobel_prob":_sobelize,
    "brightness_delta":_random_brightness,
    "contrast_delta":_random_contrast,
    "saturation_delta":_random_saturation,
    "hue_delta":_random_hue,
    "flip_left_right":_random_left_right_flip,
    "flip_up_down":_random_up_down_flip,
    "rot90":_random_rotate,
    "zoom_scale":_random_zoom,
    "mask":_random_mask
}

AUGMENT_ORDERING = ["gaussian_blur", "gaussian_noise", "brightness_delta",
                    "contrast_delta", "saturation_delta", "hue_delta",
                    "flip_left_right", "flip_up_down", "rot90", 
                    "drop_color", "sobel_prob", "zoom_scale", "mask"]



def _augment(im, aug_dict, imshape=None):
    """
    Macro to do random image augmentation
    """
    for a in AUGMENT_ORDERING:
        if a in aug_dict:
            im = SINGLE_AUG_FUNC[a](im, aug_dict[a], imshape=imshape)
    
    im = tf.clip_by_value(im, 0, 1)
    return im

def augment_function(imshape, params=True):
    """
    Define an augmentation function from a dictionary of parameters
    """
    if params == True:
        params = DEFAULT_AUGMENT_PARAMS
    @tf.function
    def _aug(x):
        return _augment(x, params, imshape=imshape)
    return _aug


