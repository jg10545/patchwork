import numpy as np
import tensorflow as tf

from patchwork.feature._contrastive import _build_negative_mask
from patchwork.feature._contrastive import _simclr_softmax_prob, _hcl_softmax_prob
from patchwork.feature._contrastive import  _contrastive_loss, _build_augment_pair_dataset


def test_build_negative_mask():
    gbs = 7
    mask = _build_negative_mask(gbs)
    assert mask.shape == (2*gbs, 2*gbs)
    assert (mask.sum(0) == 2*gbs - 2).all()


def test_simclr_softmax_prob_trivial_case():
    gbs = 7
    d = 23
    mask = _build_negative_mask(gbs)
    z = tf.nn.l2_normalize(np.random.normal(0,1, (gbs,d)))
    softmax_prob, nce_batch_acc = _simclr_softmax_prob(z,z,1,mask)

    assert softmax_prob.shape == (2*gbs,)
    assert (softmax_prob.numpy() >= 0).all()
    assert (softmax_prob.numpy() <= 1).all()
    assert nce_batch_acc.shape == ()
    assert nce_batch_acc.numpy() > 0.9



def test_simclr_softmax_prob_random_case():
    gbs = 7
    d = 23
    mask = _build_negative_mask(gbs)
    z1 = tf.nn.l2_normalize(np.random.normal(0,1, (gbs,d)))
    z2 = tf.nn.l2_normalize(np.random.normal(0,1, (gbs,d)))
    softmax_prob, nce_batch_acc = _simclr_softmax_prob(z1,z2,1,mask)

    assert softmax_prob.shape == (2*gbs,)
    assert (softmax_prob.numpy() >= 0).all()
    assert (softmax_prob.numpy() <= 1).all()
    assert nce_batch_acc.shape == ()
    assert nce_batch_acc.numpy() < 0.5




def test_contrastive_loss_simclr_case():
    gbs = 7
    d = 23
    z1 = tf.nn.l2_normalize(np.random.normal(0,1, (gbs,d)).astype(np.float32))
    z2 = tf.nn.l2_normalize(np.random.normal(0,1, (gbs,d)).astype(np.float32))
    loss, nce_batch_acc = _contrastive_loss(z1,z2,1)

    assert loss.shape == ()
    assert nce_batch_acc.shape == ()



def test_contrastive_loss_dcl_case():
    gbs = 7
    d = 23
    z1 = tf.nn.l2_normalize(np.random.normal(0,1, (gbs,d)).astype(np.float32))
    z2 = tf.nn.l2_normalize(np.random.normal(0,1, (gbs,d)).astype(np.float32))
    loss, nce_batch_acc = _contrastive_loss(z1,z2,1, decoupled=True)

    assert loss.shape == ()
    assert nce_batch_acc.shape == ()


def test_contrastive_loss_ifm_case():
    gbs = 7
    d = 23
    z1 = tf.nn.l2_normalize(np.random.normal(0,1, (gbs,d)).astype(np.float32))
    z2 = tf.nn.l2_normalize(np.random.normal(0,1, (gbs,d)).astype(np.float32))
    loss, nce_batch_acc = _contrastive_loss(z1,z2,1, eps=0.5)

    assert loss.shape == ()
    assert nce_batch_acc.shape == ()


def test_contrastive_loss_rince_case():
    gbs = 7
    d = 23
    z1 = tf.nn.l2_normalize(np.random.normal(0,1, (gbs,d)).astype(np.float32))
    z2 = tf.nn.l2_normalize(np.random.normal(0,1, (gbs,d)).astype(np.float32))
    loss, nce_batch_acc = _contrastive_loss(z1,z2,1, q=1)

    assert loss.shape == ()
    assert nce_batch_acc.shape == ()

def test_hcl_softmax_prob_trivial_case():
    gbs = 7
    d = 23
    mask = _build_negative_mask(gbs)
    z = tf.nn.l2_normalize(np.random.normal(0,1, (gbs,d)).astype(np.float32))
    softmax_prob, nce_batch_acc = _hcl_softmax_prob(z,z,1,1,0.1,mask)

    assert softmax_prob.shape == (2*gbs,)
    assert (softmax_prob.numpy() >= 0).all()
    assert (softmax_prob.numpy() <= 1).all()
    assert nce_batch_acc.shape == ()
    assert nce_batch_acc.numpy() > 0.9


def test_hcl_softmax_prob_random_case():
    gbs = 7
    d = 23
    mask = _build_negative_mask(gbs)
    z1 = tf.nn.l2_normalize(np.random.normal(0,1, (gbs,d)).astype(np.float32))
    z2 = tf.nn.l2_normalize(np.random.normal(0,1, (gbs,d)).astype(np.float32))
    softmax_prob, nce_batch_acc = _hcl_softmax_prob(z1,z2,1,1,0.1,mask)

    assert softmax_prob.shape == (2*gbs,)
    assert (softmax_prob.numpy() >= 0).all()
    assert (softmax_prob.numpy() <= 1).all()
    assert nce_batch_acc.shape == ()
    assert nce_batch_acc.numpy() < 0.5



def test_build_augment_pair_dataset(test_png_path):
    filepaths = 10*[test_png_path]
    ds = _build_augment_pair_dataset(filepaths, imshape=(32,32),
                                     batch_size=5,
                                     augment={"flip_left_right":True})
    assert isinstance(ds, tf.data.Dataset)
    for x,y in ds:
        x = x.numpy()
        y = y.numpy()
        break

    assert x.shape == (5,32,32,3)
    assert y.shape == (5,32,32,3)


def test_build_augment_pair_dataset_with_custom_dataset():
    rawdata = np.zeros((10,32,32,3)).astype(np.float32)
    ds = tf.data.Dataset.from_tensor_slices(rawdata)
    batch_size = 5
    ds = _build_augment_pair_dataset(ds, imshape=(32,32),
                              num_channels=3, norm=255,
                              augment={"flip_left_right":True},
                              single_channel=False,
                              batch_size=batch_size)
    assert isinstance(ds, tf.data.Dataset)
    for x,y in ds:
        x = x.numpy()
        y = y.numpy()
        break

    assert x.shape == (5,32,32,3)
    assert y.shape == (5,32,32,3)


def test_build_augment_pair_dataset_with_custom_pair_dataset():
    rawdata = np.zeros((10,32,32,3)).astype(np.float32)
    ds = tf.data.Dataset.from_tensor_slices((rawdata,rawdata))
    batch_size = 5
    ds = _build_augment_pair_dataset(ds, imshape=(32,32),
                              num_channels=3, norm=255,
                              augment={"flip_left_right":True},
                              single_channel=False,
                              batch_size=batch_size)
    assert isinstance(ds, tf.data.Dataset)
    for x,y in ds:
        x = x.numpy()
        y = y.numpy()
        break

    assert x.shape == (5,32,32,3)
    assert y.shape == (5,32,32,3)
