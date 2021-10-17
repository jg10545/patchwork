# -*- coding: utf-8 -*-
"""

Helper functions for SimCLR-type feature trainers

"""
import numpy as np
import tensorflow as tf

EPSILON = 1e-8

def _build_negative_mask(batch_size):
    """
    If we take the matrix of pairwise dot products of all the embeddings, it
    will include positive comparisons along the diagonal (and half-diagonals
    if the batch is symmetrized). 
    
    This function builds a (2*batch_size, 2*batch_size) matrix that masks out
    the positive cases.
    """
    mask = np.ones((2*batch_size, 2*batch_size), dtype=np.float32)
    for i in range(2*batch_size):
        for j in range(2*batch_size):
            if (i == j) or (abs(i-j) == batch_size):
                mask[i,j] = 0
    return mask


def _simclr_softmax_prob(z1, z2, temp, mask):
    """
    Compute SimCLR softmax probability for a pair of batches of normalized
    embeddings. Also compute the batch accuracy.
    
    :z1, z2: normalized embeddings (gbs,d)
    :temp: scalar Gibbs temperature parameter
    
    Returns:
        :softmax_prob:
        :nce_batch_acc:
    """
    # positive logits: just the dot products between corresponding pairs
    # shape (gbs,)
    pos = tf.reduce_sum(z1*z2, -1)/temp
    pos = tf.concat([pos,pos], 0) # from HCL code- line 38 in main.py. shape (2gbs,)
    pos_exp = tf.exp(pos)
            
    z = tf.concat([z1,z2],0)
    # negative samples- first find all pairwise combinations
    # shape (2gbs, 2gbs)
    s = tf.matmul(z, z, transpose_b=True)
    # convert negatives to logits
    neg_exp = mask*tf.exp(s/temp)     
       
    # COMPUTE BATCH ACCURACY ()
    biggest_neg = tf.reduce_max(neg_exp, -1)
    nce_batch_acc = tf.reduce_mean(tf.cast(pos_exp > biggest_neg, tf.float32))
    # COMPUTE SOFTMAX PROBABILITY (gbs,)
    softmax_prob = pos_exp/(pos_exp + tf.reduce_sum(neg_exp, -1))
    return softmax_prob, nce_batch_acc


def _hcl_softmax_prob(z1, z2, temp, beta, tau_plus, mask):
    """
    Compute HCL softmax probability for a pair of batches of normalized
    embeddings. Also compute the batch accuracy.
    
    :z1, z2: normalized embeddings (gbs,d)
    :temp: scalar Gibbs temperature parameter
    
    Returns:
        :softmax_prob:
        :nce_batch_acc:
    """
    # get the global batch size
    gbs = z1.shape[0]
    N = 2*gbs - 2
    # positive logits: just the dot products between corresponding pairs
    # shape (gbs,)
    pos = tf.reduce_sum(z1*z2, -1)/temp
    pos = tf.concat([pos,pos], 0) # from HCL code- line 38 in main.py. shape (2gbs,)
    pos_exp = tf.exp(pos)
            
    z = tf.concat([z1,z2],0)
    # negative samples- first find all pairwise combinations
    # shape (2gbs, 2gbs)
    s = tf.matmul(z, z, transpose_b=True)
    # convert negatives to logits
    neg_exp = mask*tf.exp(s/temp)     
       
    # COMPUTE BATCH ACCURACY ()
    biggest_neg = tf.reduce_max(neg_exp, -1)
    nce_batch_acc = tf.reduce_mean(tf.cast(pos_exp > biggest_neg, tf.float32))

    # importance sampling weights: shape (gbs, gbs-2)
    imp = mask*tf.exp(beta*s/temp)
    # partition function: shape (gbs,)
    Z = tf.reduce_mean(imp, -1)
    # reweighted negative logits using importance sampling: shape (gbs,)
    reweight_neg = tf.reduce_sum(imp*tf.exp(s/temp),-1)/Z
    #
    Ng = (reweight_neg - tau_plus*N*pos_exp)/(1-tau_plus)
    # constrain min value
    Ng = tf.maximum(Ng, N*tf.exp(-1/temp))
    # manually compute softmax
    softmax_prob = pos_exp/(pos_exp+Ng)
    
    
    return softmax_prob, nce_batch_acc



def _contrastive_loss(z1, z2, temp, mask, decoupled=True):
    """
    Compute contrastive loss for SimCLR or Decoupled Contrastive Learning.
    Also compute the batch accuracy.
    
    :z1, z2: normalized embeddings (gbs,d)
    :temp: scalar Gibbs temperature parameter
    
    Returns:
        :softmax_prob:
        :nce_batch_acc:
        :mask:
        :decoupled: if True, compute the weighted decoupled loss from equation 6
            of "Decoupled Contrastive Learning" by Yeh et al
    """
    # positive logits: just the dot products between corresponding pairs
    # shape (gbs,)
    pos = tf.reduce_sum(z1*z2, -1)/temp
    pos = tf.concat([pos,pos], 0) # from HCL code- line 38 in main.py. shape (2gbs,)
    pos_exp = tf.exp(pos)
            
    z = tf.concat([z1,z2],0)
    # negative samples- first find all pairwise combinations
    # shape (2gbs, 2gbs)
    s = tf.matmul(z, z, transpose_b=True)
    # convert negatives to logits
    neg_exp = mask*tf.exp(s/temp)     
       
    # COMPUTE BATCH ACCURACY ()
    biggest_neg = tf.reduce_max(neg_exp, -1)
    nce_batch_acc = tf.reduce_mean(tf.cast(pos_exp > biggest_neg, tf.float32))
    
    if decoupled:
        # compute mises-fisher weighting function (gbs,)
        weight = 2 - pos_exp/tf.reduce_mean(pos_exp)
        l_dcw = -1*weight*pos + tf.math.log(tf.reduce_sum(neg_exp, -1))
        loss = tf.reduce_mean(l_dcw)
    
    else:
        # COMPUTE SOFTMAX PROBABILITY (gbs,)
        softmax_prob = pos_exp/(pos_exp + tf.reduce_sum(neg_exp, -1))
        loss = tf.reduce_mean(-1*tf.math.log(softmax_prob + EPSILON))
        
    return loss, nce_batch_acc