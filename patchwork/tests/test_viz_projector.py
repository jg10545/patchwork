# -*- coding: utf-8 -*-
import numpy as np
from PIL import Image
from patchwork.viz._projector import _make_sprite

"""
def test_make_sprite_return_val(test_tif_path):
    imfiles = [test_tif_path]*4
    
    sprite_img = _make_sprite(imfiles, resize=[50,50])
    assert isinstance(sprite_img, Image.Image)
    assert sprite_img.width == 100
    assert sprite_img.height == 100
"""
    

def test_make_sprite_return_val(test_png_path):
    imfiles = [test_png_path]*4
    
    sprite_img = _make_sprite(imfiles, resize=[50,50])
    assert isinstance(sprite_img, Image.Image)
    assert sprite_img.width == 100
    assert sprite_img.height == 100