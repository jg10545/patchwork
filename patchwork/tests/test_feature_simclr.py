
import numpy as np
import tensorflow as tf

from patchwork.feature._simclr import _build_simclr_dataset
from patchwork.feature._simclr import _build_embedding_model
from patchwork.feature._simclr import _build_simclr_training_step


# build a tiny FCN for testing
inpt = tf.keras.layers.Input((None, None, 3))
net = tf.keras.layers.Conv2D(5,1)(inpt)
net = tf.keras.layers.MaxPool2D(10,10)(net)
fcn = tf.keras.Model(inpt, net)



def test_simclr_dataset(test_png_path):
    filepaths = 10*[test_png_path]
    batch_size = 5
    ds = _build_simclr_dataset(filepaths, imshape=(32,32),
                              num_channels=3, norm=255,
                              augment=True, single_channel=False,
                              batch_size=batch_size)
    
    assert isinstance(ds, tf.data.Dataset)
    for x,y in ds:
        break
    # since SimCLR makes augmented pairs, the batch size
    # is doubled
    assert x.shape[0] == 2*batch_size
    assert (y.numpy() == np.array([1,-1,1,-1,1,-1,1,-1,1,-1])).all()
    
def test_stratified_simclr_dataset(test_png_path, test_jpg_path):
    filepaths = 10*[test_png_path, test_jpg_path]
    labels = 10*["png", "jpg"]
    
    batch_size = 5
    ds = _build_simclr_dataset(filepaths, imshape=(32,32),
                              num_channels=3, norm=255,
                              augment={}, single_channel=False,
                              batch_size=batch_size, stratify=labels)
    
    assert isinstance(ds, tf.data.Dataset)
    for x,y in ds:
        x = x.numpy()
        break
    # since SimCLR makes augmented pairs, the batch size
    # is doubled
    assert x.shape[0] == 2*batch_size
    # since we're using a stratified dataset- each element of a particular
    # batch should be from one stratification category (which in this
    # case is an identical image)
    assert (x[0] == x[2]).all()
    
 
    
    
def test_simclr_dataset_with_custom_dataset():
    rawdata = np.zeros((10,32,32,3)).astype(np.float32)
    ds = tf.data.Dataset.from_tensor_slices(rawdata)
    batch_size = 5
    ds = _build_simclr_dataset(ds, imshape=(32,32),
                              num_channels=3, norm=255,
                              augment=True, single_channel=False,
                              batch_size=batch_size)
    
    assert isinstance(ds, tf.data.Dataset)
    for x,y in ds:
        break
    # since SimCLR makes augmented pairs, the batch size
    # is doubled
    assert x.shape[0] == 2*batch_size
    assert (y.numpy() == np.array([1,-1,1,-1,1,-1,1,-1,1,-1])).all()
    
def test_simclr_dataset_with_custom_pair_dataset():
    rawdata = np.zeros((10,32,32,3)).astype(np.float32)
    ds = tf.data.Dataset.from_tensor_slices((rawdata, rawdata))
    batch_size = 5
    ds = _build_simclr_dataset(ds, imshape=(32,32),
                              num_channels=3, norm=255,
                              augment=True, single_channel=False,
                              batch_size=batch_size)
    
    assert isinstance(ds, tf.data.Dataset)
    for x,y in ds:
        break
    # since SimCLR makes augmented pairs, the batch size
    # is doubled
    assert x.shape[0] == 2*batch_size
    assert (y.numpy() == np.array([1,-1,1,-1,1,-1,1,-1,1,-1])).all()
    
    
def test_build_embedding_model():
    model = _build_embedding_model(fcn, (32,32), 3, 17, 11)
    assert isinstance(model, tf.keras.Model)
    assert model.output_shape[-1] == 11
    assert len(model.layers) == 7 # was 8 before taking out extra batchnorm
    
    
def test_build_simclr_training_step():
    model = _build_embedding_model(fcn, (32,32), 3, 5, 7)
    opt = tf.keras.optimizers.SGD()
    step = _build_simclr_training_step(model, opt, 0.1)
    
    x = tf.zeros((4,32,32,3), dtype=tf.float32)
    y = np.array([1,-1,1,-1]).astype(np.int32)
    lossdict = step(x,y)
    
    assert isinstance(lossdict["loss"].numpy(), np.float32)
    # should include loss and average cosine similarity
    assert len(lossdict) == 2
    
    
def test_build_simclr_training_step_with_weight_decay():
    model = _build_embedding_model(fcn, (32,32), 3, 5, 7)
    opt = tf.keras.optimizers.SGD()
    step = _build_simclr_training_step(model, opt, 0.1,
                                       weight_decay=1e-6)
    
    x = tf.zeros((4,32,32,3), dtype=tf.float32)
    y = np.array([1,-1,1,-1]).astype(np.int32)
    lossdict = step(x,y)
    
    # should include total loss, crossent loss, average cosine
    # similarity and L2 norm squared
    assert len(lossdict) == 4
    