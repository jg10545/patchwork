import numpy as np
import os
from sklearn.preprocessing import normalize


#pos_categories = ["airplane", "buildings", "denseresidential", "freeway",
#                 "harbor", "intersection", "mediumresidential", "mobilehomepark",
#                 "overpass", "parkinglot", "runway", "sparseresidential", "storagetanks"]

#pos_categories = ["freeway", "intersection", "overpass", "runway"]
#pos_categories = ["mobilehomepark"]
pos_categories = ["denseresidential", "mediumresidential", "sparseresidential"]

def load_ucmerced(filelist="filepaths.txt", featfile="ucmerced_feature_vectors.numpy",
                 feat_dims=[-1,6,6,1024]):
    
    img_files = [x.strip() for x in open(filelist).readlines()]
    img_files = [x.split("data/")[-1] for x in img_files]
    
    feature_vectors = np.fromfile(featfile).reshape(-1,feat_dims[-1])
    feature_vectors = normalize(feature_vectors).reshape(*feat_dims)
    
    labels = np.zeros(len(img_files))
    for i in range(len(img_files)):
        for p in pos_categories:
            if p in img_files[i]:
                labels[i] = 1
    
    test_split = (np.arange(len(img_files)) % 10) == 0
    train_x = feature_vectors[~test_split]
    train_y = labels[~test_split]
    train_files = np.array(img_files)[~test_split]

    test_x = feature_vectors[test_split]
    test_y = labels[test_split]
    test_files = np.array(img_files)[test_split]
    
    return train_files, train_x, train_y, test_x, test_y