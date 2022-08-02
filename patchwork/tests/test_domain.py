import numpy as np
import tensorflow as tf
from patchwork._domain import _compute_mmd_loss






def test_compute_mmd_loss():
    np.random.seed(1)
    num_domains = 3
    d = 5
    N = 17

    domain_means = [np.random.normal(0, 1, size=d) for _ in range(num_domains)]
    D = np.random.randint(0, num_domains, size=N)
    X = np.stack([np.random.normal(0, 1, size=d)+domain_means[D[i]] for i in range(N)], 0).astype(np.float32)
    
    loss = _compute_mmd_loss(X,D,num_domains)
    assert loss.shape == ()
    assert loss.dtype == tf.float32
    assert loss.numpy() > 0
    
test_compute_mmd_loss()