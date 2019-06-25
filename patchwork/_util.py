import numpy as np
from osgeo import gdal

def shannon_entropy(x):
    """
    Shannon entropy of a 2D array
    """
    xprime = np.maximum(np.minimum(x, 1-1e-8), 1e-8)
    return -np.sum(xprime*np.log2(xprime), axis=1)



def tiff_to_array(f, swapaxes=True, norm_thousand=False, channels=-1):
    """
    Utility function to load a GeoTIFF to a numpy array
    """
    infile = gdal.Open(f)
    im_arr = infile.ReadAsArray()
    if swapaxes:
        im_arr = np.swapaxes(im_arr, 0, -1)
    if norm_thousand:
        im_arr = im_arr.astype(np.float32)/1000
        
    if channels > 0:
        if channels == 1:
            im_arr = im_arr[:,:,0]
        else:
            im_arr = im_arr[:,:,:channels]
        
    return im_arr