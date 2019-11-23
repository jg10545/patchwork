# -*- coding: utf-8 -*-
import os
import pytest

import tensorflow as tf


@pytest.fixture()
def test_png_path():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(this_dir, "fixtures", "test_img.png")


@pytest.fixture()
def test_tif_path():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(this_dir, "fixtures", "test_img.tif")

@pytest.fixture()
def test_geotif_path():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(this_dir, "fixtures", "test_geotiff.tif")

