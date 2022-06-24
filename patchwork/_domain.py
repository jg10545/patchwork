import numpy as np
import tensorflow as tf


def _compute_mmd_loss(features, domain_labels, num_domains, eps=1e-5):
    """
    Maximum mean discrepancy loss from the "Deep Domain Confusion" paper by Tzeng et al,
    modified for multiple domains instead of one source and one target.
    
    :features: (batch_size, d) tensor of features
    :domain_labels: (batch_size,) tensor indicating the domain of each feature
    :num_domains: int; total number of domains
    :eps: float; numerical stabilization constant for empty domain cases

    Returns the MMS loss
    """
    dense_domain_labels = tf.one_hot(domain_labels, num_domains)
    domain_counts = tf.reduce_sum(dense_domain_labels, 0, keepdims=True)
    domain_means = tf.transpose(tf.matmul(features, dense_domain_labels, transpose_a=True)/(domain_counts+eps))
    
    loss = 0
    for i in tf.range(1,num_domains):
        shuffled_means = tf.gather(domain_means, (np.arange(num_domains)+i)%num_domains)
        loss += tf.reduce_sum((domain_means-shuffled_means)**2)/num_domains
        
    return loss