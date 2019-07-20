# -*- coding: utf-8 -*-
import numpy as np
from patchwork._util import shannon_entropy, tiff_to_array, _load_img





def test_shannon_entropy():
    maxent = np.array([[0.5,0.5]])
    assert shannon_entropy(maxent)[0] == 1.0
    
    
def test_tiff_to_array(test_tif_path):
    img_arr = tiff_to_array(test_tif_path, num_channels=-1)
    
    assert isinstance(img_arr, np.ndarray)
    assert len(img_arr.shape) == 3
    assert img_arr.shape[2] == 3
    
    
def test_tiff_to_array_fixed_channels(test_tif_path):
    img_arr = tiff_to_array(test_tif_path, num_channels=2)
    
    assert isinstance(img_arr, np.ndarray)
    assert len(img_arr.shape) == 3
    assert img_arr.shape[2] == 2
    
    
def test_load_img_on_png(test_png_path):
    img_arr = _load_img(test_png_path)
    assert isinstance(img_arr, np.ndarray)
    assert len(img_arr.shape) == 3
    assert img_arr.shape[2] == 3



def test_load_img_on_png_with_resize(test_png_path):
    img_arr = _load_img(test_png_path, resize=(71,71))
    assert isinstance(img_arr, np.ndarray)
    assert len(img_arr.shape) == 3
    assert img_arr.shape == (71,71,3)    