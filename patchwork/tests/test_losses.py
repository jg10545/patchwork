# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf

from patchwork._losses import entropy_loss, masked_binary_crossentropy


def test_entropy_loss():
    # trivial probability distribution- should be close to zero entropy
    probs = np.array([[0., 1.]])
    entropy = entropy_loss(probs, probs)
    
    with tf.Session() as sess:
        entropy_computed = sess.run(entropy)
        
    assert isinstance(entropy, tf.Tensor)
    assert np.sum(entropy_computed) < 1e-5
    
    
def test_masked_binary_crossentropy():
    y_true = np.array([-1., -1, 0., 1.])
    y_pred = np.array([1., 0., 0., 1.])
    
    mbc = masked_binary_crossentropy(y_true, y_pred)
    
    with tf.Session() as sess:
        mbc_computed = sess.run(mbc)
        
    assert isinstance(mbc, tf.Tensor)
    assert np.sum(mbc_computed) < 1e-5
    