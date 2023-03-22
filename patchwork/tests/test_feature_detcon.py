# -*- coding: utf-8 -*-
import numpy as np
from PIL import Image
import tensorflow as tf
tf.random.seed(1)

from patchwork.feature._detcon_utils import _get_segments, _get_grid_segments
from patchwork.feature._detcon_utils import _segment_aug, _filter_out_bad_segments
from patchwork.feature._detcon_utils import _prepare_embeddings, _prepare_mask
from patchwork.feature._detcon import _build_segment_pair_dataset
from patchwork._tfrecord import save_dataset_to_tfrecords


def test_get_segments(test_png_path):
    num_samples = 5
    img = np.array(Image.open(test_png_path)).astype(np.float32)/255
    segs = _get_segments(img, 10, num_samples)
    assert segs.shape == (img.shape[0], img.shape[1], num_samples)


def test_get_grid_segments():
    H = 12
    W = 16
    seg = _get_grid_segments((H,W), 16)
    # check that output has correct dimensions
    assert seg.shape == (H,W,16)
    # correct that all parts of the image are
    # segmented once
    assert (seg.sum(-1) == 1).all()



def test_segment_aug_without_resize(test_png_path):
    num_samples = 5
    augment = {"flip_left_right":True, "zoom_scale":0.9, "shear":0.2}
    img = np.array(Image.open(test_png_path)).astype(np.float32)/255
    segs = _get_segments(img, 10, num_samples)

    img_aug, seg_aug = _segment_aug(img, segs, augment)
    assert img_aug.shape == img.shape
    assert seg_aug.shape == segs.shape
    # check normalization
    #totals = seg_aug.numpy().sum(0).sum(0)
    #assert np.max(np.abs(totals-1)) < 1e-4

def test_segment_aug_with_resize(test_png_path):
    num_samples = 5
    outputsize = (7,11)
    augment = {"flip_left_right":True, "zoom_scale":0.9, "shear":0.2}
    img = np.array(Image.open(test_png_path)).astype(np.float32)/255
    segs = _get_segments(img, 10, num_samples)

    img_aug, seg_aug = _segment_aug(img, segs, augment, imshape=outputsize, outputsize=outputsize)
    assert img_aug.shape == (outputsize[0], outputsize[1], img.shape[2])#img.shape
    assert seg_aug.shape == (outputsize[0], outputsize[1], segs.shape[2])
    # check normalization
    #totals = seg_aug.numpy().sum(0).sum(0)
    #assert np.max(np.abs(totals-1)) < 1e-4


def test_filter_out_bad_segments_no_empty_masks():
    seg = np.ones((1,3,5,7))
    assert _filter_out_bad_segments(None, seg, None, seg)

def test_filter_out_bad_segments_empty_masks():
    seg = np.zeros((1,3,5,7))
    assert not _filter_out_bad_segments(None, seg, None, seg)


def test_build_segment_pair_dataset(test_png_path):
    augment = {"rot90":True, "jitter":0.1}
    trainfiles = [test_png_path]*10
    ds = _build_segment_pair_dataset(trainfiles, imshape=(28,28),
                                     batch_size=5,
                                     augment=augment, mean_scale=10,
                                     outputsize=(7,7), num_samples=11)
    for x1, s1, x2, s2 in ds:
        break

    assert x1.shape == (5, 28, 28, 3)
    assert x2.shape == (5, 28, 28, 3)
    assert s1.shape == (5, 7, 7, 11)
    assert s2.shape == (5, 7, 7, 11)



def test_build_segment_pair_dataset_from_tfrecord(test_png_path, tmp_path_factory):
    augment = {"rot90":True, "jitter":0.1}
    imfiles = [test_png_path]*10

    # SAVE IT TO TFRECORD FILES
    fn = str(tmp_path_factory.mktemp("data"))
    save_dataset_to_tfrecords(imfiles, fn, num_shards=2,
                              imshape=(32,32), num_channels=3,
                              norm=255)

    ds = _build_segment_pair_dataset(fn, imshape=(32,32),
                                     batch_size=5,
                                     augment=augment, mean_scale=10,
                                     outputsize=(7,7), num_samples=11)
    for x1, s1, x2, s2 in ds:
        break

    assert x1.shape == (5, 32, 32, 3)
    assert x2.shape == (5, 32, 32, 3)
    assert s1.shape == (5, 7, 7, 11)
    assert s2.shape == (5, 7, 7, 11)



def test_prepare_embeddings():
    N = 3
    w = 5
    h = 7
    d = 11
    num_samples = 13
    x = np.zeros((N,w,h,d) ,dtype=np.float32)
    s = np.zeros((N,w,h,num_samples), dtype=np.float32)
    embeds = _prepare_embeddings(x,s)
    assert embeds.shape == (N*num_samples, d)

def test_prepare_mask():
    N = 3
    w = 5
    h = 7
    num_samples = 13
    s1 = np.zeros((N,w,h,num_samples), dtype=np.float32)
    mask = _prepare_mask(s1,s1)
    assert mask.shape == (N*num_samples, 1)
