import numpy as np
import tensorflow as tf

from patchwork._training_functions import build_training_function
from patchwork._losses import masked_binary_crossentropy

fcn = tf.keras.Sequential([
    tf.keras.layers.Conv2D(3, 1, input_shape=(5,5,3))
])

fine_tuning = tf.keras.Sequential([
    tf.keras.layers.Dense(11, input_shape=(5,5,3)),
    tf.keras.layers.GlobalAveragePooling2D()])

output = tf.keras.Sequential([
    tf.keras.layers.Dense(13, input_shape=(11,), activation="sigmoid")
])

opt = tf.keras.optimizers.SGD(1e-3)

def test_training_step_preextracted_no_semisupervised():
    batchsize = 7
    loss_fn = masked_binary_crossentropy
    xb = np.ones((batchsize,5,5,3), dtype=np.float32)
    yb = np.ones((batchsize,13), dtype=np.int64)
    # build training function
    fn = build_training_function(loss_fn, opt, fine_tuning, output)
    # run on a batch of data
    trainloss, entloss = fn(xb,yb)
    assert entloss == 0.
    assert trainloss.shape == ()
    assert trainloss.numpy() > 0.
    
    
def test_training_step_preextracted_with_semisupervised():
    batchsize = 7
    loss_fn = masked_binary_crossentropy
    xb = np.ones((batchsize,5,5,3), dtype=np.float32)
    yb = np.ones((batchsize,13), dtype=np.int64)
    
    x_wk = np.ones((2*batchsize,5,5,3), dtype=np.float32)
    x_str = 0.5*np.ones((2*batchsize,5,5,3), dtype=np.float32)
    
    # build training function
    fn = build_training_function(loss_fn, opt, fine_tuning, output,
                                 lam=1., tau=0.95)
    # run on a batch of data
    trainloss, entloss = fn(xb, yb, x_wk, x_str)
    assert entloss.shape == ()
    assert trainloss.shape == ()
    assert trainloss.numpy() > 0.
    
def test_training_step_preextracted_with_semisupervised_and_weight_decay():
    batchsize = 7
    loss_fn = masked_binary_crossentropy
    xb = np.ones((batchsize,5,5,3), dtype=np.float32)
    yb = np.ones((batchsize,13), dtype=np.int64)
    
    x_wk = np.ones((2*batchsize,5,5,3), dtype=np.float32)
    x_str = np.ones((2*batchsize,5,5,3), dtype=np.float32)
    
    # build training function
    fn = build_training_function(loss_fn, opt, fine_tuning, output,
                                 lam=1., tau=0.95, weight_decay=1e-4)
    # run on a batch of data
    trainloss, entloss = fn(xb, yb, x_wk, x_str)
    assert entloss.shape == ()
    assert trainloss.shape == ()
    assert trainloss.numpy() > 0.
    
    
    
def test_training_step_feature_extractor_no_semisupervised():
    batchsize = 7
    loss_fn = masked_binary_crossentropy
    xb = np.ones((batchsize,5,5,3), dtype=np.float32)
    yb = np.ones((batchsize,13), dtype=np.int64)
    # build training function
    fn = build_training_function(loss_fn, opt, fine_tuning, output,
                                 feature_extractor=fcn)
    # run on a batch of data
    trainloss, entloss = fn(xb,yb)
    assert trainloss.shape == ()
    assert entloss == 0.
    assert trainloss.numpy() > 0.
    
    
def test_training_step_feature_extractor_with_semisupervised():
    batchsize = 7
    loss_fn = masked_binary_crossentropy
    xb = np.ones((batchsize,5,5,3), dtype=np.float32)
    yb = np.ones((batchsize,13), dtype=np.int64)
    x_wk = np.ones((2*batchsize,5,5,3), dtype=np.float32)
    x_str = np.ones((2*batchsize,5,5,3), dtype=np.float32)
    # build training function
    fn = build_training_function(loss_fn, opt, fine_tuning, output,
                                 lam=1., tau=0.95, weight_decay=1e-4,
                                 feature_extractor=fcn)
    # run on a batch of data
    trainloss, entloss = fn(xb, yb, x_wk, x_str)
    assert entloss.shape == ()
    assert trainloss.shape == ()
    assert entloss.numpy() >= 0.
    assert trainloss.numpy() > 0.