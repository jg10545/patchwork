# -*- coding: utf-8 -*-
"""

Helper functions for SimCLR-type feature trainers

"""
import numpy as np
import tensorflow as tf
import logging

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


def _simclr_softmax_prob(z1, z2, temp, mask, eps=0):
    """
    Compute SimCLR softmax probability for a pair of batches of normalized
    embeddings. Also compute the batch accuracy.
    
    :z1, z2: normalized embeddings (gbs,d)
    :temp: scalar Gibbs temperature parameter
    :eps: epsilon parameter for implicit feature modification
    
    Returns:
        :softmax_prob:
        :nce_batch_acc:
    """
    # positive logits: just the dot products between corresponding pairs
    # shape (gbs,)
    pos = (tf.reduce_sum(z1*z2, -1) - eps)/temp
    pos = tf.concat([pos,pos], 0) # from HCL code- line 38 in main.py. shape (2gbs,)
    pos_exp = tf.exp(pos)
            
    z = tf.concat([z1,z2],0)
    # negative samples- first find all pairwise combinations
    # shape (2gbs, 2gbs)
    s = tf.matmul(z, z, transpose_b=True)
    # convert negatives to logits
    neg_exp = mask*tf.exp((s + eps)/temp)     
       
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



def _contrastive_loss(z1, z2, temp, decoupled=True, eps=0, q=0, lam=0.01):
    """
    Compute contrastive loss for SimCLR or Decoupled Contrastive Learning.
    Also compute the batch accuracy.
    
    :z1, z2: normalized embeddings (gbs,d)
    :temp: scalar Gibbs temperature parameter
    :decoupled: bool; whether to use the loss function from the DCL paper
    :eps: epsilon parameter from the IFM paper
    :q: q parameter for RINCE loss in "Robust Contrastive Learning against Noisy Views"
    :lam: lambda parameter for RINCE loss in "Robust Contrastive Learning against Noisy Views"
    
    Returns:
        :softmax_prob:
        :nce_batch_acc:
        :decoupled: if True, compute the weighted decoupled loss from equation 6
            of "Decoupled Contrastive Learning" by Yeh et al
    """
    if decoupled & (eps > 0):
        logging.warn("mixing decoupled contrastive learning with implicit feature modification? you're out of control, man.")
    if decoupled & (q>0):
        logging.error("you can have RINCE or DCL but not both")
    if (eps > 0)&(q > 0):
        logging.warn("mixing IFM and RINCE seems weird but i'm not going to stop you.")
    # construct mask
    mask = tf.stop_gradient(_build_negative_mask(z1.shape[0]))
    # positive logits: just the dot products between corresponding pairs
    # shape (gbs,)
    pos = (tf.reduce_sum(z1*z2, -1) - eps)/temp
    pos = tf.concat([pos,pos], 0) # from HCL code- line 38 in main.py. shape (2gbs,)
    pos_exp = tf.exp(pos)
            
    z = tf.concat([z1,z2],0)
    # negative samples- first find all pairwise combinations
    # shape (2gbs, 2gbs)
    s = tf.matmul(z, z, transpose_b=True)
    # convert negatives to logits
    neg_exps = mask*tf.exp((s + eps)/temp)     
    neg_exp = tf.reduce_sum(neg_exps, -1)
    
    # COMPUTE BATCH ACCURACY ()
    biggest_neg = tf.reduce_max(neg_exps, -1)
    nce_batch_acc = tf.reduce_mean(tf.cast(pos_exp > biggest_neg, tf.float32))
    
    
    if decoupled:
        logging.info("using decoupled contrastive learning objective")
        # compute mises-fisher weighting function (gbs,)
        sigma = 0.5 # section 4.2 of paper used this value for one set of experiments
        factor = tf.exp(tf.reduce_sum(z1*z2, -1)/sigma) # (gbs,)
        factor = tf.concat([factor, factor], 0) # (2gbs,)
        weight = tf.stop_gradient(2 - factor/tf.reduce_mean(factor)) # (2gbs,)
        l_dcw = -1*weight*pos + tf.math.log(neg_exp)
        loss = tf.reduce_mean(l_dcw)
    
    else:
        if q == 0:  
            logging.info("using standard contrastive learning objective")
            # COMPUTE SOFTMAX PROBABILITY (gbs,)
            softmax_prob = pos_exp/(pos_exp + neg_exp)
            loss = tf.reduce_mean(-1*tf.math.log(softmax_prob + EPSILON))
        else:
            logging.info("using RINCE learning objective")
            loss = tf.reduce_mean(-1*(pos_exp**q)/q + (lam*(pos_exp + neg_exp)**q)/q)
        
    return loss, nce_batch_acc