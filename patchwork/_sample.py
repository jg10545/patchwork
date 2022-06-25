# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import tensorflow as tf

PROTECTED_COLUMN_NAMES = ["filepath", "exclude", "viewpath", "validation", "subset"]


def find_unlabeled(df):
    """
    Return boolean series of totally unlabeled data points
    """
    label_types = [x for x in df.columns if 
                   x not in PROTECTED_COLUMN_NAMES]
    return pd.isnull(df[label_types]).values.prod(axis=1).astype(bool)

def find_fully_labeled(df):
    """
    Return boolean series of totally labeled data points
    """
    label_types = [x for x in df.columns if 
                   x not in PROTECTED_COLUMN_NAMES]
    return pd.notnull(df[label_types]).values.prod(axis=1).astype(bool)

def find_partially_labeled(df):
    """
    Return boolean series of partially labeled data points
    """
    return (~find_unlabeled(df))&(~find_fully_labeled(df))

def find_labeled_indices(df):
    # return indices of fully labeled records
    #unlabeled = find_unlabeled(df)
    labeled = find_fully_labeled(df)
    return np.arange(len(labeled))[labeled]

def find_excluded_indices(df):
    # return indices of excluded records
    excluded = df["exclude"] == True
    return np.arange(len(df))[excluded]

def find_subset(df, label_status, exclude_status, s):
    """
    Macro to return a Boolean Series defining a subset of a dataframe
    
    :df: the dataframe
    :label_status: "unlabeled", "partial", or "labeled"
    :exclude_status: "not excluded", "excluded", or "validation"
    :s: string; how to subset it. values could be:
        -all
        -unlabeled X (for class X)
        -contains X (for class X)
        -doesn't contain X (for class X)
        -subset Y (for subsetting category Y)
    """
    if label_status == "unlabeled":
        lab = find_unlabeled(df)
    elif label_status == "partial":
        lab = find_partially_labeled(df)
    elif label_status == "labeled":
        lab = find_fully_labeled(df)
    else:
        assert False, f"i don't recognize label_status '{label_status}'"
        
    if exclude_status == "not excluded":
        ex = df["exclude"] != True
    elif exclude_status == "excluded":
        ex = df["exclude"] == True
    elif exclude_status == "validation":
        ex = df["validation"] == True

    
    if s == "all":
        return lab & ex
    elif "contains" in s:
        s = s.replace("contains:", "").strip()
        return lab & ex & (df[s] == 1)
    elif "doesn't contain" in s:
        s = s.replace("doesn't contain:", "").strip()
        return lab & ex & (df[s] == 0)
    elif "subset" in s:
        s = s.replace("subset:", "").strip()
        return lab & ex & (df["subset"].astype(str) == s)
    elif "unlabeled" in s:
        s = s.replace("unlabeled: ", "")
        return lab & ex & pd.isnull(df[s])
    else:
        assert False, "sorry can't help you"
        


def _prepare_df_for_stratified_sampling(df):
    """
    Take a label dataframe and return a list of arrays for every 
    combination of class and 0,1 value. 
    
    So a dataframe containing subset for classes "foo" and "bar"
    should generate a list with four elements; the non-excluded non-validation
    training points where foo=1, where foo=0, where bar=1, and where bar=0.
    
    Each array contains the indices of the matching rows of the dataframe
    """
    not_excluded = (df["exclude"] != True)&(df["validation"] != True)
    categories = [x for x in df.columns if x not in PROTECTED_COLUMN_NAMES]
    indexlist = []
    
    for c in categories:
        for l in [0,1]:
            subset = not_excluded&(df[c] == l)
            if subset.sum() > 0:
                indexlist.append(df[subset].index.values)
    return indexlist


def stratified_sample(df, N=1000, return_indices=False, sampling="class",
                      indexlist=None):
    """
    Build a stratified sample from a dataset. Maps NAs in partially
    labeled records to -1.
    
    :df: DataFrame containing file paths (in a "filepath" column) and
        labels in other columns
    :N: number of samples
    :return_indices: whether to return indices instead of file paths
    :sampling: string; method to use to choose which subset to sample
        from. can be "class", "instance", or "squareroot"
            -"class": each class/label sampled with equal probability
            -"instance": each data point samples with equal probability
            -"squareroot": class/label chosen in proportion to the square
                root of the number of examples
    :indexlist: optional; precompute indices for stratification
    
    Returns
    (filepaths or indices), label vectors
    """
    if indexlist is None:
        indexlist = _prepare_df_for_stratified_sampling(df)
    categories = [x for x in df.columns if x not in PROTECTED_COLUMN_NAMES]
    num_cats = len(categories)
    
    # find sampling probabilities
    if sampling == "class":
        p = np.ones(len(indexlist))
    elif sampling == "instance":
        p = np.array([float(len(f)) for f in indexlist])
    elif sampling == "squareroot":
        p = np.array([np.sqrt(len(f)) for f in indexlist])
    else:
        assert False, "I don't know what to do with sampling == '%s'"%sampling
    # normalize
    p /= p.sum()
    
    # now do the actual sampling
    inds = np.zeros(N, dtype=int)
    sampchoice = np.arange(len(indexlist))
    for n in range(N):
        # choose which list to sample from
        i = np.random.choice(sampchoice, p=p)
        # pick an index
        j = np.random.choice(indexlist[i])
        inds[n] = j
    # get a new dataframe where each row is a sample   
    sampled = df.loc[inds,:]
    # labels
    ys = sampled[categories].values
    # map missing values to -1
    ys[np.isnan(ys)] = -1
    # return labels with filepaths or indices
    if return_indices:
        return inds, ys
    else:
        return sampled.filepath.values, ys




def stratified_subset_sample(df, N=1000):
    """
    Build a sample from a dataset, stratified by the "subset" column
    
    :df: DataFrame containing file paths (in a "filepath" column) and
        labels in other columns
    :N: number of samples
    
    Returns
    (filepaths), label vectors
    """
    not_excluded = (df["exclude"] != True)&(df["validation"] != True)
    assert "subset" in df.columns, "can't stratify by subset if there's no subset"
    subsets = df.subset.unique()
    
    indexlist = []
    for s in subsets:
        sub = not_excluded&(df["subset"] == s)
        indexlist.append(df[sub].index.values)
    
    inds = np.zeros(N, dtype=int)
    ys = np.random.choice(np.arange(len(indexlist)), size=N)
    for n in range(N):
        # pick an index
        inds[n] = np.random.choice(indexlist[ys[n]])
    # get the associated filepaths
    sampled = df["filepath"].loc[inds].values
    
    return sampled, ys

def unlabeled_sample(df, N=1000):
    """
    Build a sample of unlabeled records from a dataset
    
    :df: DataFrame containing file paths (in a "filepath" column) and
        labels in other columns
    """
    unlabeled = df["filepath"][find_unlabeled(df)].values
    return np.random.choice(unlabeled, size=N, replace=True)
    
def _build_in_memory_dataset(features, indices, labels, batch_size=16, unlabeled_indices=None):
    """
    Build tf.data.Dataset object for training in the pre-extracted feature case.
    
    :features: rank-4 tensor of pre-extracted features
    :indices: indices of features generated by stratified sampler
    :labels: label vectors corresponding to indices
    :unlabeled_indices: indices of unlabeled features for semi-supervised learning
    """
    x = features[indices].astype(np.float32)
    if unlabeled_indices is not None:
        unlabeled_samp_indices = np.random.choice(unlabeled_indices, replace=True, size=len(indices))

        x_unlab = features[unlabeled_samp_indices].astype(np.float32)
        ds = tf.data.Dataset.from_tensor_slices(((x,labels), x_unlab))
    else:
        ds = tf.data.Dataset.from_tensor_slices((x, labels))
    ds = ds.batch(batch_size)
    ds = ds.prefetch(1)
    return ds