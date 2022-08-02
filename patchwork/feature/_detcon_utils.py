# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
import warnings
import skimage.segmentation

from patchwork._augment import SINGLE_AUG_FUNC


SEG_AUG_FUNCTIONS = ["flip_left_right", "flip_up_down", "rot90", "shear", "zoom_scale", "center_zoom_scale"]


def _get_segments(img, mean_scale=1000, num_samples=16, return_enough_segments=False):
    """
    Wrapper for computing segments for an image. Inputs an image tensor and
    returns a stack of binary segmentation masks.

    :img: (H,W,C) image tensor
    :mean_scale: average scale parameter for Felzenszwalb's algorithm. Actual
        value will be sampled from (0.5*mean_scale, 1.5*mean_scale), and min_size
        will be set to the scale.
    :num_samples: number of segments to compute. if more segments are found than
        num_samples, they'll be uniformly subsampled without replacement; if fewer
        are found they'll be sampled with replacement.
    :return_enough_segments: whether to include a Boolean indicator of whether
    """
    # randomly choose the segmentation scale
    scale = np.random.uniform(0.5*mean_scale, 1.5*mean_scale)
    # run heuristic segmentation
    segments = skimage.segmentation.felzenszwalb(img, scale=scale,
                                                 min_size=int(scale))
    # sample a set of segmentations to use; bias toward larger ones
    max_segment = segments.max()
    indices = np.arange(max_segment+1)
    seg_count = np.array([(segments == i).sum()+1 for i in indices])
    p = seg_count/seg_count.sum()
    # try this for error correction?
    if num_samples <= max_segment:
        sampled_indices = np.random.choice(indices, p=p, size=num_samples,
                                           replace=False)
    else:
        warnings.warn("not enough unique segments; sampling WITH replacement")
        sampled_indices = np.random.choice(indices, size=num_samples, replace=True)
    # build normalized segment occupancy masks for each segment we choose
    seg_tensor = np.stack([(segments == i)/seg_count[i] for i in sampled_indices],
                          -1).astype(np.float32)

    if return_enough_segments:
        enough_segs = num_samples <= max_segment
        return seg_tensor, enough_segs
    return seg_tensor

def _get_grid_segments(imshape, num_samples=16):
    gridsize = int(np.sqrt(num_samples))
    h,w = imshape
    seg = np.zeros((h,w, gridsize**2), dtype=np.int64)

    index = 0
    dy = int(h/gridsize)
    dx = int(w/gridsize)
    for i in range(gridsize):
        for j in range(gridsize):
            seg[i*dy:(i+1)*dy, j*dx:(j+1)*dx,index] = 1
            index += 1
    return seg

def _segment_aug(img, seg, aug, imshape=None, outputsize=None):
    """
    """
    num_channels = img.shape[-1]
    if imshape is None:
        imshape = (img.shape[0], img.shape[1])
    x = tf.concat([img, tf.cast(seg, tf.float32)], -1)
    for f in SEG_AUG_FUNCTIONS:
        if f in aug:
            x = SINGLE_AUG_FUNC[f](x, aug[f], imshape=imshape, num_channels=x.shape[-1])

    img_aug = x[:,:,:num_channels]
    seg_aug = x[:,:,num_channels:]

    if outputsize is not None:
        seg_aug = tf.image.resize(seg_aug, outputsize, method="area")
    # normalize segments
    norm = tf.expand_dims(tf.expand_dims(tf.reduce_sum(seg_aug, [0,1]),0),0)
    seg_aug /= (norm+1e-8)

    return img_aug, seg_aug

def _filter_out_bad_segments(img1, seg1, img2, seg2):
    """
    It's possible for shearing or scaling augmentation to sample
    one segment completely out of the image- use this function
    to filter out those cases
    """
    minval = tf.reduce_min(tf.reduce_sum(seg1, [0,1])*tf.reduce_sum(seg2, [0,1]))
    if minval < 0.5:
        warnings.warn("filtering bad segment")
        return False
    else:
        return True


def _prepare_embeddings(h, m):
    """
    Combine FCN outputs with segmentation masks to build a batch of
    mask-pooled hidden vectors. Represents the calculation of h_{m}
    in the first equation in Henaff et al's paper

    :h: batch of embeddings; (N,w,h,d)
    :m: batch of NORMALIZED segmentation tensors; (N,w,h,num_samples)

    Returns a tensor of shape (N*num_samples, d)
    """
    d = h.shape[-1]
    h = tf.expand_dims(h, 4)
    m = tf.expand_dims(m, 3)
    hm = tf.reduce_mean(h*m, [1,2])
    return tf.reshape(hm, [-1, d])


def _prepare_mask(m1, m2):
    """
    :m1, m2: masks of segmentation tensors; (N,w,h,num_samples)

    Returns a mask of shape (N*num_samples, 1)
    """
    m1_sum = tf.reduce_sum(m1, [1,2])
    m2_sum = tf.reduce_sum(m2, [1,2])
    return tf.reshape(m1_sum*m2_sum, [-1,1])
