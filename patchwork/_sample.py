# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

def find_unlabeled(df):
    """
    Return boolean series of totally unlabeled data points
    """
    label_types = [x for x in df.columns if 
                   x not in ["filepath", "exclude", "viewpath"]]
    return pd.isnull(df[label_types]).values.prod(axis=1).astype(bool)

def find_fully_labeled(df):
    """
    Return boolean series of totally labeled data points
    """
    label_types = [x for x in df.columns if 
                   x not in ["filepath", "exclude"]]
    return pd.notnull(df[label_types]).values.prod(axis=1).astype(bool)

def find_partially_labeled(df):
    """
    Return boolean series of partially labeled data points
    """
    return (~find_unlabeled(df))&(~find_fully_labeled(df))


def find_subset(df, s):
    """
    Macro to return a Boolean Series defining a subset of a dataframe
    
    :df: the dataframe
    :s: string; how to subset it. values could be:
        -unlabeled
        -fully labeled
        -partially labeled
        -excluded
        -not excluded
        -unlabeled X (for class X)
        -contains X (for class X)
        -doesn't contain X (for class X)
    """
    if s == "unlabeled":
        return find_unlabeled(df)
    elif s == "fully labeled":
        return find_fully_labeled(df)
    elif s == "partially labeled":
        return find_partially_labeled(df)
    elif s == "excluded":
        return df["excluded"] == 1
    elif s == "not excluded":
        return df["excluded"] == 0
    elif "unlabeled:" in s:
        s = s.replace("unlabeled:", "").strip()
        return pd.isnull(df[s])
    elif "contains" in s:
        s = s.replace("contains:", "").strip()
        return df[s] == 1
    elif "doesn't contain" in s:
        s = s.replace("doesn't contain:", "").strip()
        return df[s] == 0
    else:
        assert False, "sorry can't help you"



def stratified_sample(df, N=1000):
    """
    Build a stratified sample from a dataset. Maps NAs in partially
    labeled records to -1.
    
    :df: DataFrame containing file paths (in a "filepath" column) and
        labels in other columns
    """
    #if N is None:
    #    N = len(df)
    index = df.index.values
    filepaths = df["filepath"].values
    label_types = [x for x in df.columns if x not in ["filepath", "exclude"]]

    file_lists = [[
            index[df[l] == 0] \
            for l in label_types if (df[l] == 0).sum() > 0
            ],
            [
            index[df[l] == 1] \
            for l in label_types if (df[l] == 1).sum() > 0
            ]]
    num_lists = [len(file_lists[0]), len(file_lists[1])]
    
    outlist = []
    ys = []
    for n in range(N):
        z = np.random.choice([0,1])
        i = np.random.choice(np.arange(num_lists[z]))
        outlist.append(filepaths[np.random.choice(file_lists[z][i])])
        #y_vector = label_df.loc[z].values.astype(float)
        y_vector = df[label_types].loc[z].values.astype(float)
        y_vector[np.isnan(y_vector)] = -1
        ys.append(y_vector.astype(int))
        
    return outlist, np.stack(ys)


def unlabeled_sample(df, N=1000):
    """
    Build a sample of unlabeled records from a dataset
    
    :df: DataFrame containing file paths (in a "filepath" column) and
        labels in other columns
    """
    unlabeled = df["filepath"][find_unlabeled(df)].values
    return np.random.choice(unlabeled, size=N, replace=True)
    
    