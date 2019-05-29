import numpy as np
import pandas as pd
import warnings
import panel as pn

from patchwork._labeler import Labeler
from patchwork._modelpicker import ModelPicker
from patchwork._trainmanager import TrainManager

prompt_txt = "Enter comma-delimited list of class-1 patches:"
EPSILON = 1e-5

def sample_batch_indices_and_labels(df, classes, batch_size):
    """
    Stratified sampler for batch generation
    
    :df:
    :classes:
    :batch_size:
    """
    num_classes = len(classes)
    # find the labeled indices for each category
    labeled_indices = [df[df["label"] == c].index.to_numpy() for c in classes]
    # select the number of examples from each category
    choices_per_class = np.random.multinomial(batch_size, np.ones(num_classes)/num_classes)
    # randomly select (with replacement) that many examples from each category
    batch_indices = np.concatenate([np.random.choice(l, c, replace=True)
                              for (l,c) in zip(labeled_indices, choices_per_class)])
    # assemble the corresponding labels
    batch_labels = np.concatenate([i*np.ones(choices_per_class[i]) for i in range(num_classes)])
    
    return batch_indices, batch_labels



class PatchWork(object):
    
    def __init__(self, feature_vecs, df, classes, model=None,
                 dim=3, imsize=100):
        """
        :feature_vecs: numpy array of feature data for each unlabeled training point
        :df: pandas DataFrame containing a "filepath" column and optionally a
            "label" column
        :classes: list of strings containing class nams
        :epochs: how many epochs to train for each iteration
        :model: a Keras model
        :dim: grid dimension for labeler- show a (dim x dim) square of images
        :imsize: pixel size of images in labeler
        """
        self.feature_vecs = feature_vecs
        self.df = df
        if "label" not in df.columns:
            self.df["label"] = None
        
        self.classes = classes
        self.model = model
        self.N = feature_vecs.shape[0]
        assert self.N == len(df), "Size of feature array should match dataframe"
        
        # BUILD THE GUI
        # initialize Labeler object
        self.labeler = Labeler(classes, df, dim, imsize)
        # initialize model picker
        self.modelpicker = ModelPicker(num_classes=len(classes),
                                       inpt_channels=feature_vecs.shape[-1])
        # make a train manager- pass this object to it
        self.trainmanager = TrainManager(self)
        
        
        
    def _update_unlabeled(self):
        """
        update our array keeping track of unlabeled images
        """
        self.unlabeled_indices = np.arange(self.N)[np.isnan(self.labels)]
        

       
    def _training_generator(self, bs):
        """
        Generator for labeled data. Takes care of stratified sampling
        across classes.
        """
        while True:
            indices, labels = sample_batch_indices_and_labels(self.df, 
                                                    self.classes, bs)
            yield self.feature_vecs[indices,:], labels

        
    def panel(self):
        """
        
        """
        return pn.Tabs(("Model", self.modelpicker), 
                       ("Train", self.trainmanager.panel()), 
                       ("Annotate", self.labeler.panel()))
        
        