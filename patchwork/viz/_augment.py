# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from patchwork._util import _load_img
from patchwork._augment import augment_function


def _single_augplot(filepath, aug_params=True, norm=255, num_channels=3, resize=None):
    """
    Input a path to an image and an augmentation function; sample
    15 augmentations and display using matplotlib.
    """
    img = _load_img(filepath, norm=norm, num_channels=num_channels, resize=resize)
    aug_func = augment_function(img.shape[:2], aug_params)
    plt.figure()
    plt.subplot(4,4,1)
    plt.imshow(img)
    plt.axis(False)
    plt.title("original")
    for i in range(2,17):
        plt.subplot(4,4,i)
        plt.imshow(aug_func(img).numpy())
        plt.axis(False)
        
        
        
def _multiple_augplot(filepaths, aug_params=True, num_resamps=5, 
                      norm=255, num_channels=3, resize=None):
    """
    
    """
    num_files = len(filepaths)
    img = _load_img(filepaths[0], norm=norm, num_channels=num_channels, resize=resize)
    aug_func = augment_function(img.shape[:2], aug_params)
    
    plt.figure()
    for i in range(num_files):
        img = _load_img(filepaths[i], norm=norm, num_channels=num_channels, resize=resize)
        plt.subplot(num_files+1, num_resamps+1, 1+i*(num_resamps+1))
        plt.imshow(img)
        plt.axis(False)
        if i == 0: plt.title("original")
        for j in range(num_resamps):
            plt.subplot(num_files+1, num_resamps+1, j+2+i*(num_resamps+1))
            plt.imshow(aug_func(img).numpy())
            plt.axis(False)
        
        
def augplot(filepath, aug_params=True, norm=255, num_channels=3, resize=None):
    """
    Pass either a path to an image or a list of paths and a dictionary of
    augmentation parameters- and visualize random augmentations on the image(s).
    
    If the list has more than 10 images, 10 will be randomly selected.
    
    :filepath: string containing a path to an image file, or a list of paths
    :aug_params: dictionary of augmentation parameters (or True)
    """
    if isinstance(filepath, str):
        _single_augplot(filepath, aug_params, norm, num_channels, resize)
    else:
        if len(filepath) <= 5:
            _multiple_augplot(filepath, aug_params, norm=norm, 
                              num_channels=num_channels, resize=resize)
        else:
            _multiple_augplot([str(x) for x in np.random.choice(filepath, size=5, replace=False)], 
                              aug_params, norm=norm, 
                              num_channels=num_channels, resize=resize)
        
        
        
        
        
        
        
        
        