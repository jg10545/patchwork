
import numpy as np
import tensorflow as tf
tf.random.set_seed(1)

from patchwork.feature._simclr import _build_embedding_model
from patchwork.feature._simclr import _build_simclr_training_step


# build a tiny FCN for testing
inpt = tf.keras.layers.Input((None, None, 3))
net = tf.keras.layers.Conv2D(5,1)(inpt)
net = tf.keras.layers.MaxPool2D(10,10)(net)
fcn = tf.keras.Model(inpt, net)

    
    
def test_build_embedding_model():
    model = _build_embedding_model(fcn, (32,32), 3, 17, 11)
    assert isinstance(model, tf.keras.Model)
    assert model.output_shape[-1] == 11
    assert len(model.layers) == 8
    
    
def test_build_embedding_model_extra_layers():
    model = _build_embedding_model(fcn, (32,32), 3, 17, 11, num_projection_layers=3)
    assert isinstance(model, tf.keras.Model)
    assert model.output_shape[-1] == 11
    assert len(model.layers) == 8+3 # extra dense, batchnorm, and activation
    
    
def test_build_simclr_training_step():
    model = _build_embedding_model(fcn, (32,32), 3, 5, 7)
    opt = tf.keras.optimizers.SGD()
    step = _build_simclr_training_step(model, opt, 0.1)
    
    x = tf.zeros((4,32,32,3), dtype=tf.float32)
    #y = np.array([1,-1,1,-1]).astype(np.int32)
    y = x
    lossdict = step(x,y)
    
    assert isinstance(lossdict["loss"].numpy(), np.float32)
    # should include loss and average cosine similarity
    assert len(lossdict) == 4 #2
    
    
def test_build_embedding_model_directclr():
    model = _build_embedding_model(fcn, (32,32), 3, 0, 7)
    opt = tf.keras.optimizers.SGD()
    step = _build_simclr_training_step(model, opt, 0.1)
    
    x = tf.zeros((4,32,32,3), dtype=tf.float32)
    #y = np.array([1,-1,1,-1]).astype(np.int32)
    y = x
    lossdict = step(x,y)
    
    assert isinstance(lossdict["loss"].numpy(), np.float32)
    # should include loss and average cosine similarity
    assert len(lossdict) == 4 #2
    
    
def test_build_simclr_training_step_with_weight_decay():
    model = _build_embedding_model(fcn, (32,32), 3, 5, 7)
    opt = tf.keras.optimizers.SGD()
    step = _build_simclr_training_step(model, opt, 0.1,
                                       weight_decay=1e-6)
    
    x = tf.zeros((4,32,32,3), dtype=tf.float32)
    #y = np.array([1,-1,1,-1]).astype(np.int32)
    y = x
    lossdict = step(x,y)
    
    # should include total loss, crossent loss, average cosine
    # similarity and L2 norm squared
    assert len(lossdict) == 4
    