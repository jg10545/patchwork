import numpy as np
import tensorflow as tf

from patchwork.feature._iic import compute_mutual_information, compute_p



def test_compute_mutual_information():
    testarr = np.array([1.,0.,0.]).reshape((1,-1))
    results = compute_mutual_information(testarr, testarr)
    
    assert len(results) == 3
    for r in results:
        assert isinstance(r.numpy(), float)
        
        
def test_compute_p():
    testarr = np.array([1.,0.,0.]).astype(np.float32).reshape((1,-1))
    head = tf.keras.layers.Dense(5)
    P = compute_p(testarr, testarr, head)
    
    assert P.shape == (1,5,5,1)
    assert P.max() == 1.