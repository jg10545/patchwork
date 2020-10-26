import numpy as np

from patchwork.viz._projector import _make_sprite

def _make_kernel_sprites(fcn):
    """
    Input a fully-convolutional network; generate an image showing a grid of
    (independently normalized) views of the first-level convolution kernels
    """
    # get the kernel weights for the first conv layer
    layers = [l for l in fcn.layers if len(l.trainable_variables) > 0]
    weight = [x for x in layers[0].trainable_variables if "kernel" in x.name][0].numpy()
    # independently normalize each
    kernels = []
    for i in range(weight.shape[-1]):
        w = weight[:,:,:,i]
        w /= (w.max() - w.min())
        w -= w.min()
        kernels.append(w)    
    kernels = np.stack(kernels, 0)
    return np.array(_make_sprite(kernels, spritesize=kernels.shape[1]))
    