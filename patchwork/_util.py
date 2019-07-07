import numpy as np
from PIL import Image
from osgeo import gdal

def shannon_entropy(x):
    """
    Shannon entropy of a 2D array
    """
    xprime = np.maximum(np.minimum(x, 1-1e-8), 1e-8)
    return -np.sum(xprime*np.log2(xprime), axis=1)



def tiff_to_array(f, swapaxes=True, norm=255, channels=-1):
    """
    Utility function to load a TIFF or GeoTIFF file to a numpy array,
    wrapping gdal.
    
    :f: string; path to file
    :swapaxes: expect that the image will be read to a (channels, height, width)
        array that needs to be reordered to (height, width, channels)
    :norm: if not None, cast raster to float and divide by this
    :channels: if -1, loadall channels; otherwise load the specified number 
        (e.g. 1 or 3 for display)
    """
    infile = gdal.Open(f)
    im_arr = infile.ReadAsArray()
    if swapaxes:
        im_arr = np.swapaxes(im_arr, 0, -1)
    if norm is not None:
        im_arr = im_arr.astype(np.float32)/norm
        
    if channels > 0:
        if channels == 1:
            im_arr = im_arr[:,:,0]
        else:
            im_arr = im_arr[:,:,:channels]
        
    return im_arr


def _load_img(f, norm=255, channels=3, resize=None):
    """
    Generic image-file-to-numpy-array loader. Uses filename
    to choose whether to use PIL or GDAL.
    
    :f: string; path to file
    :norm: value to normalize images by
    :channels: number of input channels (for GDAL only)
    :resize: pass a tuple to resize to
    """
    if type(f) is not str:
        f = f.numpy().decode("utf-8") 
    if ".tif" in f:
        img_arr = tiff_to_array(f, swapaxes=True, 
                             norm=norm, channels=channels)
    else:
        img = Image.open(f)
        img_arr = np.array(img).astype(np.float32)/norm
        
    if resize is not None and channels ==3:
        img_arr = np.array(Image.fromarray(
                (255*np.swapaxes(img_arr, 0,1)).astype(np.uint8)).resize(resize)
                    ).astype(np.float32)/norm
    return img_arr