# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
from patchwork._util import _load_img
from patchwork._augment import augment_function


def augplot(filepath, aug_params=True):
    """
    Input a path to an image and an augmentation function; sample
    15 augmentations and display using matplotlib.
    """
    img = _load_img(filepath)
    aug_func = augment_function(aug_params)
    plt.subplot(4,4,1)
    plt.imshow(img)
    plt.axis(False)
    plt.title("original")
    for i in range(2,17):
        plt.subplot(4,4,i)
        plt.imshow(aug_func(img).numpy())
        plt.axis(False);