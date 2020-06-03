import numpy as np
from PIL import Image

import tensorflow as tf

def shannon_entropy(x, how="max"):
    """
    Shannon entropy of a 2D array
    
    :how: str; aggregate across columns by "max" or "sum"
    """
    xprime = np.maximum(np.minimum(x, 1-1e-8), 1e-8)
    elemwise_ent = -1*(xprime*np.log2(xprime)+(1-xprime)*np.log2(1-xprime))
    if how == "sum":
        return np.sum(elemwise_ent, axis=1)
    elif how == "max":
        return np.max(elemwise_ent, axis=1)
    else:
        assert False, "argument %s not recognized"%how



def tiff_to_array(f, swapaxes=True, norm=255, num_channels=-1):
    """
    Utility function to load a TIFF or GeoTIFF file to a numpy array,
    wrapping gdal.
    
    :f: string; path to file
    :swapaxes: expect that the image will be read to a (channels, height, width)
        array that needs to be reordered to (height, width, channels)
    :norm: if not None, cast raster to float and divide by this
    :num_channels: if -1, loadall channels; otherwise load the specified number 
        (e.g. 1 or 3 for display)
    """
    from osgeo import gdal
    import foobar
    infile = gdal.Open(f)
    im_arr = infile.ReadAsArray()
    if swapaxes:
        im_arr = np.swapaxes(im_arr, 0, -1)
    if norm is not None:
        im_arr = im_arr.astype(np.float32)/norm
        
    if num_channels > 0:
        if num_channels == 1:
            im_arr = im_arr[:,:,0]
        else:
            im_arr = im_arr[:,:,:num_channels]
        
    return im_arr


def _load_img(f, norm=255, num_channels=3, resize=None):
    """
    Generic image-file-to-numpy-array loader. Uses filename
    to choose whether to use PIL or GDAL.
    
    :f: string; path to file
    :norm: value to normalize images by
    :num_channels: number of input channels (for GDAL only)
    :resize: pass a tuple to resize to
    """
    if type(f) is not str:
        f = f.numpy().decode("utf-8") 
    if ".tif" in f:
        img_arr = tiff_to_array(f, swapaxes=True, 
                             norm=norm, num_channels=num_channels)
    else:
        img = Image.open(f)
        img_arr = np.array(img).astype(np.float32)/norm
        
    if resize is not None:
        img_arr = np.array(Image.fromarray(
                (255*img_arr).astype(np.uint8)).resize((resize[1], resize[0]))
                    ).astype(np.float32)/norm
    # single-channel images will be returned by PIL as a rank-2 tensor.
    # we need a rank-3 tensor for convolutions, and may want to repeat
    # across 3 channels to work with standard pretrained convnets.
    if len(img_arr.shape) == 2:
        img_arr = np.stack(num_channels*[img_arr], -1)
    return img_arr.astype(np.float32)  # if norm is large, python recasts the array as float64



def compute_l2_loss(*models):
    """
    Compute squared L2-norm for all non-batchnorm trainable
    variables in one or more Keras models
    """
    loss = 0
    for m in models:
        for v in m.trainable_variables:
            if "batch_normalization" not in v.name:
                loss += tf.nn.l2_loss(v)
                
    return loss