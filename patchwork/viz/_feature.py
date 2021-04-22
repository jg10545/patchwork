# -*- coding: utf-8 -*-
import numpy as np
from PIL import Image
from scipy.spatial.distance import cdist


def _build_query_image(features, labels, labeldict, num_returned=4):
    """
    For a label dictionary and set of associated features- build a
    numpy array that we can use as a tensorboard image that shows one
    example image for each category, and the [num_returned] images with
    closest vectors (by cosine similarity)
    
    :features: (num_labels, d) numpy array of features
    :labels:
    :labeldict: dictionary mapping filepaths to categories
    :num_returned: number of similar images to show per category
    """
    filepaths = list(labeldict.keys())
    indices = np.arange(labels.shape[0])
    # pull the index of the first example from each category
    query_indices = np.array([indices[labels == l][0] for l in set(labels)])
    # and get the vector associated with those examples: [num_categories, d]
    query_vecs = features[query_indices]
    # find pairwise cosine distances from each query to all vectors [num_categories, len(labeldict)]
    distances = cdist(query_vecs, features, metric="cosine")
    # find the first num_returned+1 neared images to each query (the first should be the 
    # query itself)
    returned_indices = []
    for i in range(distances.shape[0]):
        returned_indices.append(distances[i].argsort()[0:num_returned+1])
    # now load up those images and concatenate into a figure
    rows = []
    for i in range(len(returned_indices)):
        row = []
        for j in returned_indices[i]:
            row.append(np.array(Image.open(filepaths[j])))
        rows.append(np.concatenate(row,1))
    final = np.concatenate(rows, 0)
    return np.expand_dims(final, 0)