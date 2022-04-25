# -*- coding: utf-8 -*-
"""
Integrated gradients- taken from https://www.tensorflow.org/tutorials/interpretability/integrated_gradients with some small modifications
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image


from patchwork._augment import augment_function


def _interpolate_images(baseline, image, alphas):
    alphas_x = alphas[:, tf.newaxis, tf.newaxis, tf.newaxis]
    baseline_x = tf.expand_dims(baseline, axis=0)
    input_x = tf.expand_dims(image, axis=0)
    delta = input_x - baseline_x
    images = baseline_x +  alphas_x * delta
    return images



def _compute_gradients(images, model, aug_embeds, neg_embeds=None):
    """
    
    """
    with tf.GradientTape() as tape:
        tape.watch(images)
        embeds = model(images, training=False)
        # embeds: (num_alphas, d)
        embeds = tf.nn.l2_normalize(embeds, 1)
        # aug_embeds: (num_aug, d)
        aug_embeds = tf.nn.l2_normalize(aug_embeds, 1)
        # product: num_alphas, num_aug
        loss = -1*tf.reduce_mean(
            tf.reduce_sum(tf.matmul(embeds, aug_embeds, transpose_b=True), 1))
        if neg_embeds is not None:
            neg_embeds = tf.nn.l2_normalize(neg_embeds, 1)
            loss += tf.reduce_mean(
            tf.reduce_sum(tf.matmul(embeds, neg_embeds, transpose_b=True), 1))
    return tape.gradient(loss, images)


def _integral_approximation(gradients):
    # riemann_trapezoidal
    grads = (gradients[:-1] + gradients[1:]) / tf.constant(2.0)
    integrated_gradients = tf.math.reduce_mean(grads, axis=0)
    return integrated_gradients


def _wrap_fcn(fcn, num_channels=3):
    """
    Input a fully-convolutional network as a keras object and return 
    a model with average-pooled features
    """
    inpt = tf.keras.layers.Input((None, None, num_channels))
    net = fcn(inpt)
    net = tf.keras.layers.GlobalAvgPool2D()(net)
    return tf.keras.Model(inpt, net)


def _generate_augmented_embeddings(img_arr, model, imshape, aug_params, num_augments=25):
    """
    
    """
    # generate a batch of augmented images
    aug_func = augment_function(imshape, aug_params)
    augmented_ims = tf.stack([aug_func(img_arr) for _ in range(num_augments)], 0)
    # run through the model
    return model(augmented_ims, training=False)

def _generate_negative_embeddings(neg_ims, model, imshape, norm=255):
    """
    
    """
    # load and stack a batch of negative images
    if isinstance(neg_ims[0], str):
        neg_ims = [np.array(Image.open(f).resize(imshape).astype(np.float32))/norm for 
           f in neg_ims]
    neg_ims = np.stack(neg_ims, 0)
    # run through the model
    return model(neg_ims, training=False)



def integrated_gradients(img, fcn, aug_params, imshape=(256,256), num_channels=3, norm=255,
                        num_augments=25, num_alphas=50, neg_images=None):
    """
    
    """
    # load image if an array wasn't passed
    if isinstance(img, str):
        img = np.array(Image.open(img).resize(imshape)).astype(np.float32)/norm
    
    # prepare stack of interpolated images for integration
    baseline = tf.zeros_like(img)
    alphas = tf.linspace(start=0.0, stop=1.0, num=num_alphas)
    interpolated_images = _interpolate_images(baseline, img, alphas)
    
    # wrap model to get average-pooled features
    model = _wrap_fcn(fcn, num_channels=num_channels)
    
    # generate embeddings from augmented image
    aug_embeds = _generate_augmented_embeddings(img, model, imshape, aug_params, num_augments)
    
    # if negative images were passed, get embeddings for those too
    if neg_images is not None:
        neg_images = _generate_negative_embeddings(neg_images, model, imshape, norm)
    
    # get gradients
    grads = _compute_gradients(interpolated_images, model, aug_embeds, neg_images)
    
    # integrate them
    integrated_grads = _integral_approximation(grads)
    attribution = tf.reduce_sum(tf.math.abs(integrated_grads), axis=-1)
    
    # draw a figure
    fig, axs = plt.subplots(nrows=1, ncols=3)

    axs[0].set_title('Original image')
    axs[0].imshow(img)
    axs[0].axis('off')

    axs[1].set_title('Attribution mask')
    axs[1].imshow(attribution)
    axs[1].axis('off')

    axs[2].set_title('Overlay')
    axs[2].imshow(img)
    axs[2].contour(attribution, levels=3)
    axs[2].axis('off');
