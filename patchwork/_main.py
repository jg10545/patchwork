import numpy as np
import pandas as pd
import panel as pn
import tensorflow as tf

from patchwork._labeler import Labeler
from patchwork._modelpicker import ModelPicker
from patchwork._trainmanager import TrainManager
from patchwork._sample import stratified_sample, find_unlabeled, _build_in_memory_dataset
from patchwork.loaders import dataset
from patchwork._losses import entropy_loss, masked_binary_crossentropy
from patchwork._util import _load_img

EPSILON = 1e-5



class PatchWork(object):
    
    def __init__(self, df, feature_vecs=None, feature_extractor=None, classes=[],
                 imshape=(256,256), num_channels=3, norm=255,
                 num_parallel_calls=2, logdir=None, aug=True, dim=3):
        """
        Initialize either with a set of feature vectors or a feature extractor
        
        :df: pandas DataFrame containing a "filepath" column and optionally a
            "label" column
        :feature_vecs: numpy array of feature data for each unlabeled training point
        :feature_extractor: keras Model object- should be frozen
        :classes: list of strings containing class names
        :imshape: pixel size to reshape image to
        :num_channels: number of channels per image
        :norm: value to divide image data by to scale it to the unit interval. This will usually be 255.
        :num_parallel_calls: parallel processes for image loading
        :outfile: file path to save labels to during annotation
        :aug: Boolean or dict of augmentation parameters. Only matters if you're
            using a feature extractor instead of static features.
        :dim: grid dimension for labeler- show a (dim x dim) square of images
        """
        self.fine_tuning_model = None
        # by default, pandas maps empty values to np.nan. in case the user
        # is passing saved labels in, replace those with None
        self.df = df.replace({pd.np.nan: None})
        self.feature_vecs = feature_vecs
        self.feature_extractor = feature_extractor
        self._aug = aug


        self._imshape = imshape
        self._norm = norm
        self._num_channels = num_channels
        self._num_parallel_calls = num_parallel_calls
        self._semi_supervised = False
        self._logdir = logdir
        self.models = {"feature_extractor":feature_extractor}
        
        if "exclude" not in df.columns:
            df["exclude"] = False
            
        for c in classes:
            if c not in df.columns:
                df[c] = None
        self.classes = [x for x in df.columns if x not in ["filepath", "exclude", "viewpath", "validation"]]
        # initialize dataframe of predictions
        self.pred_df = pd.DataFrame(
                {c:np.random.uniform(0,1,len(df)) for c in self.classes},
                index=df.index)
        
        
        # BUILD THE GUI
        # initialize Labeler object
        self.labeler = Labeler(self.classes, self.df, self.pred_df, self._load_img,
                               dim=dim, logdir=logdir)
        # initialize model picker
        if self.feature_vecs is not None:
            inpt_channels = self.feature_vecs.shape[-1]
        else:
            inpt_channels = self.feature_extractor.output.get_shape().as_list()[-1]

        self.modelpicker = ModelPicker(len(self.classes), inpt_channels, self,
                                       feature_extractor=feature_extractor)
        # make a train manager- pass this object to it
        self.trainmanager = TrainManager(self)
        
        
        
    def _update_unlabeled(self):
        """
        update our array keeping track of unlabeled images
        """
        self.unlabeled_indices = np.arange(self.N)[np.isnan(self.labels)]
      
        
    def panel(self):
        """
        
        """
        return pn.Tabs(("Model", self.modelpicker.panel()), 
                       ("Train", self.trainmanager.panel()), 
                       ("Annotate", self.labeler.panel()))
        
    
    def _training_dataset(self, batch_size=32, num_samples=None):
        """
        Build a single-epoch training set
        """
        if num_samples is None:
            num_samples = len(self.df)
        # LIVE FEATURE EXTRACTOR CASE
        if self.feature_vecs is None:
            files, ys = stratified_sample(self.df, num_samples)
            unlab_fps = None
            if self._semi_supervised:
                unlabeled_filepaths = self.df.filepath.values[find_unlabeled(self.df)]
                
                unlab_fps = np.random.choice(unlabeled_filepaths,
                                             replace=True, size=num_samples)
            return dataset(files, ys, imshape=self._imshape, 
                       num_channels=self._num_channels,
                       num_parallel_calls=self._num_parallel_calls, 
                       batch_size=batch_size,
                       augment=self._aug, unlab_fps=unlab_fps)[0]
        # PRE-EXTRACTED FEATURE CASE
        else:
            inds, ys = stratified_sample(self.df, num_samples, return_indices=True)
            if self._semi_supervised:
                unlabeled_indices = np.arange(len(self.df))[find_unlabeled(self.df)]
            else:
                unlabeled_indices = None
                
            return _build_in_memory_dataset(self.feature_vecs, 
                                          inds, ys, batch_size=batch_size,
                                          unlabeled_indices=unlabeled_indices)

    def _pred_dataset(self, batch_size=32):
        """
        Build a dataset for predictions
        """
        num_steps = int(np.ceil(len(self.df)/batch_size))
        if self.feature_vecs is None:
            files = self.df["filepath"].values
            return dataset(files, imshape=self._imshape, 
                       num_channels=self._num_channels,
                       num_parallel_calls=self._num_parallel_calls, 
                       batch_size=batch_size, shuffle=False,
                       augment=False)#, num_steps
        # PRE-EXTRACTED FEATURE CASE
        else:
            return tf.data.Dataset.from_tensor_slices(self.feature_vecs
                                                      ).batch(batch_size), num_steps
    
    
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

            
            

    def _run_one_training_epoch(self, batch_size=32, num_samples=None):
        """
        Run one training epoch
        """
        ds = self._training_dataset(batch_size, num_samples)
        
        if self._semi_supervised:
            for (x, x_unlab), y in ds:
                loss, ss_loss = self._training_function(x, y, self._opt, x_unlab)
                self.training_loss.append(loss.numpy())
                self.semisup_loss.append(ss_loss.numpy())
        else:
            for x, y in ds:
                loss, ss_loss = self._training_function(x, y, self._opt)
                self.training_loss.append(loss.numpy())
                self.semisup_loss.append(ss_loss.numpy())
                
    
    def fit(self, batch_size=32, num_samples=None):
        """
        Run one training epoch
        """
        if self.feature_vecs is not None:
            x, y = self._training_dataset(batch_size, num_samples)
            return self._training_model.fit(x, y, batch_size=batch_size)
        else:
            dataset, num_steps = self._training_dataset(batch_size, num_samples)
            return self._training_model.fit(dataset, steps_per_epoch=num_steps, epochs=1)
    
    def predict_on_all(self, batch_size=32):
        """
        Run inference on all the data; save to self.pred_df
        """
        ds, num_steps = self._pred_dataset(batch_size)
        predictions = self.models["full"].predict(ds, steps=num_steps)
        #if self.feature_vecs is not None:
        #    predictions = self.model.predict(self.feature_vecs, 
        #                                     batch_size=batch_size)
        #else:
        #    dataset, num_steps = self._pred_dataset(batch_size)
        #    predictions = self.model.predict(dataset, steps=num_steps)
        self.pred_df.loc[:, self.classes] = predictions
    
    def _stratified_sample(self, N=None):
        return stratified_sample(self.df, N)
    
    def _load_img(self, f):
        """
        Wrapper for patchwork._util._load_img
        
        :f: string; path to file
        """
        return _load_img(f, norm=self._norm, num_channels=self._num_channels, 
                resize=self._imshape)
    
    
    
    
    
    
    
    
    