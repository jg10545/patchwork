# -*- coding: utf-8 -*-
import numpy as np



def stratified_sample(df, N=None):
    """
    Build a stratified sample from a dataset.
    
    :df: DataFrame containing file paths (in a "filepath" column) and
        labels in other columns
    """
    if N is None:
        N = len(df)
    index = df.index.values
    filepaths = df["filepath"].values
    # dataframe of just labels
    label_df = df.drop("filepath", 1)
    # list of label types
    label_types = list(label_df.columns)
    
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
        y_vector = label_df.loc[z].values.astype(float)
        y_vector[np.isnan(y_vector)] = -1
        ys.append(y_vector.astype(int))
        
    return outlist, np.stack(ys)