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
    # return indices of fully/partially labeled records
    unlabeled = find_unlabeled(df)
    return np.arange(len(unlabeled))[~unlabeled]

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
        



def stratified_sample(df, N=1000, return_indices=False):
    """
    Build a stratified sample from a dataset. Maps NAs in partially
    labeled records to -1.
    
    :df: DataFrame containing file paths (in a "filepath" column) and
        labels in other columns
    :N: number of samples
    :return_indices: whether to return indices instead of file paths
    
    Returns
    (filepaths or indices), label vectors
    """
    index = df.index.values
    # figure out which records aren't excluded, either by being tagged
    # "exclude" or tagged "validation"
    not_excluded = (df["exclude"] != True)&(df["validation"] != True)
    filepaths = df["filepath"].values
    label_types = [x for x in df.columns if x not in PROTECTED_COLUMN_NAMES]
    
    # build a hierarchical set of lists for sampling:
    # outer list has two elements: negative and positive labels
    # each of those lists has one list per class, so long as that class has
    # the right type of labels
    # each element of those is an array of indices meeting the class/label criteria
    file_lists = [[
            index[(df[l] == 0)&not_excluded] \
            for l in label_types if (df[l][not_excluded] == 0).sum() > 0
            ],
            [
            index[(df[l] == 1)&not_excluded] \
            for l in label_types if (df[l][not_excluded] == 1).sum() > 0
            ]]
    num_lists = [len(file_lists[0]), len(file_lists[1])]
    
    outlist = []
    ys = []
    inds = []
    for n in range(N):
        # choose to sample a positive or negative label
        z = np.random.choice([0,1])
        # choose a category with positive/negative
        i = np.random.choice(np.arange(num_lists[z]))
        # choose an index consistent with i and z
        ind = np.random.choice(file_lists[z][i])
        inds.append(ind)
        outlist.append(filepaths[ind])
        # convert labels to a vector with None mapped to -1
        y_vector = df[label_types].loc[ind].values.astype(float)
        y_vector[np.isnan(y_vector)] = -1
        ys.append(y_vector.astype(int))
        
    if return_indices:
        return np.array(inds), np.stack(ys)
    else:
        return outlist, np.stack(ys)


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