import numpy as np
import pandas as pd
import warnings
import panel as pn
import tensorflow as tf

from patchwork._labeler import Labeler
from patchwork._modelpicker import ModelPicker
from patchwork._trainmanager import TrainManager
from patchwork._sample import stratified_sample
from patchwork._loader import dataset
from patchwork._losses import entropy_loss, masked_binary_crossentropy

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
    
    def __init__(self, df, feature_vecs=None, feature_extractor=None, classes=[],
                 dim=3, imsize=100, imshape=(256,256), num_channels=3,
                 num_parallel_calls=2):
        """
        Initialize either with a set of feature vectors or a feature extractor
        
        :df: pandas DataFrame containing a "filepath" column and optionally a
            "label" column
        :feature_vecs: numpy array of feature data for each unlabeled training point
        :feature_extractor: keras Model object- should be frozen
        :classes: list of strings containing class names
        :dim: grid dimension for labeler- show a (dim x dim) square of images
        :imsize: pixel size of images in labeler
        """
        self.fine_tuning_model = None
        self.df = df
        self.feature_vecs = feature_vecs
        self.feature_extractor = feature_extractor
        if feature_extractor is not None:
            if feature_extractor.trainable == True:
                warnings.warn("Feature extractor wasn't frozen- was this on purpose?")
        self._imshape = imshape
        self._num_channels = num_channels
        self._num_parallel_calls = num_parallel_calls
        self._semi_supervised = False
        
        for c in classes:
            if c not in df.columns:
                df[c] = None
        self.classes = [x for x in df.columns if x != "filepath"]
        
        
        # BUILD THE GUI
        # initialize Labeler object
        self.labeler = Labeler(classes, df, dim, imsize)
        # initialize model picker
        self.modelpicker = ModelPicker(num_classes=len(self.classes),
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
        assert False, "DEPRECATEEEED"
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
        
    
    def _training_dataset(self, batch_size=32):
        """
        Build a single-epoch training set
        """
        files, ys = stratified_sample(self.df, len(self.df))
        return dataset(files, ys, imshape=self._imshape, 
                       num_channels=self._num_channels,
                       num_parallel_calls=self._num_parallel_calls, 
                       batch_size=batch_size,
                       augment=True)

    
    def _pred_dataset(self, batch_size=32):
        return dataset(self.df["filepath"].values, imshape=self._imshape, 
                       num_channels=self._num_channels,
                       num_parallel_calls=self._num_parallel_calls, 
                       batch_size=batch_size,
                       augment=False)
    
    def build_model(self, entropy_reg=0):
        """
        Sets up a Keras model in self.model
        
        NOTE the semisup case will have different predict outputs. maybe
        separate self.model from a self.training_model?
        """
        opt = tf.keras.optimizers.RMSprop(1e-3)

        # JUST A FINE-TUNING NETWORK
        if self.feature_vecs is not None:
            inpt_shape = self.feature_vecs.shape[1:]
            inpt = tf.keras.layers.Input(inpt_shape)
            output = self.fine_tuning_model(inpt)
            self.model = tf.keras.Model(inpt, output)
            
        elif self.feature_extractor is not None:
            inpt_shape = [self._imshape[0], self._imshape[1], self._num_channels]
            # FEATURE EXTRACTOR + FINE-TUNING NETWORK
            inpt = tf.keras.layers.Input(inpt_shape)
            features = self.feature_extractor(inpt)
            output = self.fine_tuning_model(features)
            self.model = tf.keras.Model(inpt, output)

        else:
            assert False, "i don't know what you want from me"
            
        # semi-supervised learning or not? 
        if entropy_reg > 0:
            self._semi_supervised = True
            unlabeled_inpt = tf.keras.layers.Input(inpt_shape)
            unlabeled_output = self.model(unlabeled_inpt)
                
            self._training_model = tf.keras.Model([inpt, unlabeled_inpt],
                                            [output, unlabeled_output])
            self._training_model.compile(opt, 
                                loss=[masked_binary_crossentropy, entropy_loss],
                                loss_weights=[1, entropy_reg])
        else:
            self._semi_supervised = False
            self._training_model = self.model
            self._training_model.compile(opt, loss=masked_binary_crossentropy)

            
            
            
    
    def fit(self, batch_size=32):
        """
        Run one training epoch
        """
        if self.feature_vecs is not None:
            self._training_model.fit("foo", batch_size=batch_size)
        else:
            dataset, num_steps = self._training_dataset(batch_size)
            self._training_model.fit(dataset, steps_per_epoch=num_steps, epochs=1)
    
    def predict_on_all(self, batch_size=32):
        """
        Run inference on all the data; save to self.pred_df
        """
        dataset, num_steps = self._pred_dataset(batch_size)
        predictions = self.model.predict(dataset, steps_per_epoch=num_steps, epochs=1)
    
    def _stratified_sample(self, N=None):
        return stratified_sample(self.df, N)
    
    
    
    
    
    
    
    
    
    