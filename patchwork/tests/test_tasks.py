# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
from PIL import Image

from patchwork._tasks import _rotate


def test_rotate(test_png_path):
    img = np.array(Image.open(test_png_path)).astype(np.float32)/255
    img_r, theta = _rotate(img)
    
    img_r = img_r.numpy()
    assert img.shape == img_r.shape
    for i in range(3):
        assert (img[:,:,i].sum() - img_r[:,:,i].sum())**2 < 1e-2
        
    assert theta.numpy() in [0,1,2,3]
    
