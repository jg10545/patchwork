# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import backend as K
from sklearn.preprocessing import LabelEncoder
import warnings
import os
from tqdm import tqdm
import yaml

from patchwork._losses import masked_sparse_categorical_crossentropy
from patchwork.loaders import _image_file_dataset
from patchwork._augment import augment_function
from patchwork._layers import _next_layer

INPUT_PARAMS = ["imshape", "num_channels", "norm", "batch_size",
                "shuffle", "num_parallel_calls", "sobel", "single_channel"]

def _encode_classes(train, val):
    """
    Encode a pandas Series into an array of integers. Empty
    values are mapped to -1.
    """
    series = pd.concat([train,val])
    notnull = pd.notnull(series)
    
    # build label encoder
    le = LabelEncoder()
    le.fit(series[notnull])
    # encode both train and validation series
    train_indices = -1*np.ones(len(train))
    train_nn = pd.notnull(train)
    train_indices[train_nn] = le.transform(train[train_nn])
    
    val_indices = -1*np.ones(len(val))
    val_nn = pd.notnull(val)
    val_indices[val_nn] = le.transform(val[val_nn])
    
    # let's build in a check while we're at it
    for c in le.classes_:
        if c not in train.values:
            warnings.warn("Class %s missing from training set"%c)
        if c not in val.values:
            warnings.warn("Class %s missing from validation set"%c)
    
    return train_indices, val_indices, le.classes_



def _dataframe_to_classes(train, val, tasks, filepath="filepath"):
    """
    Parse a multitask-labeled dataframe into the data structures
    we'll need.
    
    :train: pandas dataframe of training labels- one column for file locations
        and one per task
    :val: pandas dataframe of validation labels; same format as train
    :task: list of strings; names of task columns
    :filepath: string; name of column containing file locations
    
    Returns
    :outdict: dictionary containing all the train and validation data structures
    :class_dict: dictionary keeping track of the possible values for each class
    """
    class_dict = {}
    train_indices = []
    val_indices = []
    
    for c in tasks:
        t_ind, v_ind, classes = _encode_classes(train[c], val[c])
        class_dict[c] = classes
        train_indices.append(t_ind)
        val_indices.append(v_ind)
    
    train_indices = np.stack(train_indices, -1)
    val_indices = np.stack(val_indices, -1)
    outdict = {"train_files":train[filepath].values,
              "val_files":val[filepath].values,
              "train_indices":train_indices,
              "val_indices":val_indices}
    return outdict, class_dict



        
        
def _assemble_full_network(fcn, task_dimensions, shared_layers=[], 
                          task_layers=[128,"p",128],
                         train_fcn=False, global_pooling="max",
                           **kwargs):
    """
    Macro for generating a multihead keras network
    
    :fcn: keras Model object; fully-convolutional network to use as feature extractor
    :task_dimensions: list of ints; number of classes per task
    :shared layers: list specifying 0 or more layers after the feature extractor,
        before the task heads
    :task_layers: list specifying 0 or more hidden layers to use for each head
    :train_fcn: Boolean; whether to update fcn weights during training
    :global_pooling: "max" or "average"; how to pool task features for classification
    :**kwargs: passed to the _next_layer() function
    """
    outputs = []
    # set up shared layers
    inpt = fcn.input
    shared = fcn.output
    
    for s in shared_layers:
        shared = _next_layer(shared, s, **kwargs)
        
    # set up output head networks
    for d in task_dimensions:
        net = shared
        for s in task_layers:
            net = _next_layer(net, s, **kwargs) 
        if global_pooling == "max":
            net = tf.keras.layers.GlobalMaxPool2D()(net)
        elif global_pooling == "average":
            net = tf.keras.layers.GlobalAvgPool2D()(net)
        else:
            assert False, "don't know how to pool your model"
        final = tf.keras.layers.Dense(d, activation="softmax")
        outputs.append(final(net))
        
    shared_model = tf.keras.Model(inpt, shared)
    full_model = tf.keras.Model(inpt, outputs)
    
    # get a list of variables to optimize during training- if we're freezing
    # the feature extractor, leave those out
    trainvars = full_model.trainable_variables
    if not train_fcn:
        fcn_varnames = [v.name for v in fcn.trainable_variables]
        trainvars = [v for v in trainvars if v.name not in fcn_varnames]
        
    return {"fcn":fcn, "shared":shared_model,
           "full":full_model}, trainvars
            
            
            
def _build_multitask_training_step(model, trainvars, optimizer, 
                                  task_loss_weights, adaptive=False):
    """
    
    """
    @tf.function
    def train_step(x, y):
        with tf.GradientTape() as tape:
            loss = 0
            task_losses = []
            outputs = model(x, training=True)
            
            for pred, y_true, weight in zip(outputs, y, 
                                            task_loss_weights):
                task_loss = masked_sparse_categorical_crossentropy(y_true, pred)
                task_losses.append(task_loss)
                if adaptive:
                    # interpret weight as log(sigma^2). Kendall's paper mentions
                    # that they use this as it's more numerically stable
                    #sig_sq = tf.math.exp(weight)
                    inv_sig_sq = tf.math.exp(-1*weight)
                    #loss += task_loss/(sig_sq + K.epsilon()) + 0.5*weight
                    loss += task_loss*inv_sig_sq + 0.5*weight
                else:
                    loss += weight*task_loss
                
        gradients = tape.gradient(loss, trainvars)
        optimizer.apply_gradients(zip(gradients, trainvars))
        return loss, task_losses
    return train_step    
            
def _sampling_probabilities(indices):
    """
    Compute marginal sampling probabilities for each training point so
    that the classes will be balanced.
    """
    N, k = indices.shape
    probs = np.zeros(N)
    
    for i in range(k):
        not_missing = indices[:,i] != -1
        probs[not_missing] += 1./(np.sum(not_missing)*k)
    return probs

         
def _mtdataset(filepaths, labels, imshape, num_parallel_calls, norm,
               num_channels, single_channel, aug, batch_size):
    ds = _image_file_dataset(filepaths, imshape, num_parallel_calls,
                                 norm, num_channels,
                                 single_channel=single_channel)
    if aug:
        _aug = augment_function(imshape, aug)
        ds = ds.map(_aug)
    
    if labels is not None:    
        label_ds = [tf.data.Dataset.from_tensor_slices(labels[:,i]) for i in 
                                                    range(labels.shape[1])]
        ds = tf.data.Dataset.zip((ds, *label_ds))    
    ds = ds.batch(batch_size)
    return ds




class MultiTaskTrainer(object):
    """
    Class for managing training of a multitask convnet.
    
    Expects training and validation data to be a pandas DataFrame
    with one column containing paths to files, and one categorical 
    column per task.
    
    The shared_layers and task_layer kwargs use a shorthand for defining
    part of a convnet- each inputs a list with one element per layer:
        -if the layer is an integer: add a 3x3 convolution with that many
            filters, ReLU activation, and same padding
        -if the layer is "p": add a 2x2 max pooling layer
        -if the layer is "d": add a 2D spatial dropout layer with prob=0.5
        -if the layer is "r": add a residual convolutional layer
        
    If you set the kwarg task_weights="adaptive", the trainer will use the noise
    model from "Multi-Task Learning Using Uncertainty to Weigh Losses for Scene
    Geometry and Semantics" by Kendall et al (2018) to try to adaptively balance
    the weights.
    """
    
    
    def __init__(self, logdir, trainingdata, valdata, tasks, fcn,
                 filepaths="filepath", task_weights=None,
                 shared_layers=[], task_layers=[128,"p",128], 
                 train_fcn=False,
                 lr=1e-3, lr_decay=0, balance_probs=True,
                 augment=False, imshape=(256,256), num_channels=3,
                 norm=255, batch_size=64, shuffle=True, num_parallel_calls=None,
                 single_channel=False, notes=""):
        """
        :logdir: (string) path to log directory
        :trainingdata: pandas dataframe of training data
        :valdata: pandas dataframe of validation data
        :tasks: list of strings- columns in train/val representing the 
            different tasks
        :fcn: (keras Model) 
        :filepaths: string; name of column containing file path data
        :task_weights: list of weights for each task. Pass None for equal
            weights, or "adaptive" to do uncertainty weighing
        :shared_layers: specify shared layers downstream of the feature extractor
        :task_layers: specify layer structure for each task head, downstream
            of the shared layers
        :train_fcn: (bool) whether to let the feature extractor train
        :lr: learning rate
        :lr_decay: learning rate decay (set to 0 to disable)
        :balance_probs: if True, bias sampling during training to attempt to
            make sure each task is represented. Does not correct for class 
            imbalance within tasks.
        :augment: (dict) dictionary of augmentation parameters, True for defaults or
            False to disable augmentation
        :extractor_param: kwarg for extractor
        :imshape: (tuple) image dimensions in H,W
        :num_channels: (int) number of image channels
        :norm: (int or float) normalization constant for images (for rescaling to
               unit interval)
        :batch_size: (int) batch size for training
        :shuffle: (bool) whether to shuffle training set
        :num_parallel_calls: (int) number of threads for loader mapping
        :sobel: whether to replace the input image with its sobel edges
        :single_channel: if True, expect a single-channel input image and 
            stack it num_channels times.
        :notes: any experimental notes you want recorded in the config.yml file
        """
        adaptive = task_weights == "adaptive"
        self.logdir = logdir
        self._aug = augment
        self._tasks = tasks
        labels, class_dict = _dataframe_to_classes(trainingdata, valdata,
                                                   tasks, filepath=filepaths)
        self._labels = labels
        if balance_probs:
            probs = _sampling_probabilities(self._labels["train_indices"])
        else:
            probs = np.ones(len(trainingdata))/len(trainingdata)
        self._labels["probs"] = probs
        
        self._class_dict = class_dict
        task_dimensions = [len(class_dict[t]) for t in tasks]
        
        models, trainvars = _assemble_full_network(fcn, task_dimensions, shared_layers,
                                              task_layers=task_layers, 
                                              train_fcn=train_fcn, 
                                              global_pooling="max")
        self._models = models
        
        # create optimizer
        if lr_decay > 0:
            learnrate = tf.keras.optimizers.schedules.ExponentialDecay(lr, 
                                            decay_steps=lr_decay, decay_rate=0.5,
                                            staircase=False)
        else:
            learnrate = lr
        self._optimizer = tf.keras.optimizers.Adam(learnrate)
        
        if task_weights is None:
            task_weights = [1 for _ in tasks]
        elif adaptive:
            task_weights = [tf.Variable(1., dtype=tf.float32, 
                                        name="weight_%s"%t) for t in tasks]
            trainvars += task_weights
        self._task_weights = task_weights
        self._training_step = _build_multitask_training_step(self._models["full"], 
                                                             trainvars, 
                                                             self._optimizer,
                                                             task_weights,
                                                             adaptive)
        # build validation dataset. 
        self._val_ds = _mtdataset(self._labels["val_files"], None,
                                  imshape, num_parallel_calls, norm, num_channels,
                                  single_channel, False, batch_size)
        
        self._file_writer = tf.summary.create_file_writer(logdir, flush_millis=10000)
        self._file_writer.set_as_default()
        self.step = 0
        
        self._parse_configs(augment=augment, filepaths=filepaths, 
                            adaptive=adaptive, shared_layers=shared_layers,
                            task_layers=task_layers, train_fcn=train_fcn,
                            lr=lr, 
                            lr_decay=lr_decay, imshape=imshape, 
                            num_channels=num_channels,
                            norm=norm, batch_size=batch_size, shuffle=shuffle,
                            num_parallel_calls=num_parallel_calls,
                            single_channel=single_channel, notes=notes)
        
        
        
    def _parse_configs(self, **kwargs):
        """
        Organize input parameters and save to a YAML file so you can
        find them later.
        """
        self.config = {}
        self.input_config = {}
        self.augment_config = False
        
        for k in kwargs:
            if k == "augment":
                self.augment_config = kwargs[k]
            elif k in INPUT_PARAMS:
                self.input_config[k] = kwargs[k]
            else:
                self.config[k] = kwargs[k]
                
        config_path = os.path.join(self.logdir, "config.yml")
        config_dict = {"model":self.config, "input":self.input_config, 
                       "augment":self.augment_config}
        yaml.dump(config_dict, open(config_path, "w"), default_flow_style=False)

    
    def fit(self, epochs=1, save=True, evaluate=True):
        """
        Train the feature extractor
        
        :epochs: number of epochs to train for
        :save: if True, save after each epoch
        :evaluate: if True, run eval metrics after each epoch
        """
        N = len(self._labels["train_files"])
        for e in tqdm(range(epochs)):
            # resample labels each epoch
            indices = np.random.choice(np.arange(N), p=self._labels["probs"],
                                       size=N, replace=True)
            ds = _mtdataset(self._labels["train_files"][indices], 
                            self._labels["train_indices"][indices],
                            self.input_config["imshape"],
                            self.input_config["num_parallel_calls"],
                            self.input_config["norm"],
                            self.input_config["num_channels"],
                            self.input_config["single_channel"],
                            self.augment_config,
                            self.input_config["batch_size"])
            
            for x, *y in ds:
                loss, task_losses = self._training_step(x,y)

                self._record_scalars(total_loss=loss)
                self._record_scalars(**{"loss_"+x:t for x,t in 
                                        zip(self._tasks, task_losses)})
                
                self.step += 1
            
            if save:
                self.save()
            if evaluate:
                self.evaluate()
    
    def save(self):
        """
        Write model(s) to disk
        
        Note: tried to use SavedModel format for this and got a memory leak;
        think it's related to https://github.com/tensorflow/tensorflow/issues/32234
        
        For now sticking with HDF5
        """
        for m in self._models:
            path = os.path.join(self.logdir, m+".h5")
            self._models[m].save(path, overwrite=True, save_format="h5")
            
    def evaluate(self):
        predictions = self._models["full"].predict(self._val_ds)
        
        for i in range(len(self._tasks)):
            task = self._tasks[i]
            preds = predictions[i]
            class_preds = preds.argmax(axis=1)
            y_true = self._labels["val_indices"][:,i]
            mask = tf.cast(y_true != -1, tf.float32)
            norm = tf.reduce_sum(mask) + K.epsilon()
            
            acc = tf.reduce_sum(tf.cast(class_preds == y_true, tf.float32)*mask)/norm
            self._record_scalars(**{"val_accuracy_%s"%task:acc})
            
            self._record_scalars(**{"weight_%s"%t:w for t,w in
                                    zip(self._tasks, self._task_weights)})
            
                
            
    def _record_scalars(self, **scalars):
        for s in scalars:
            tf.summary.scalar(s, scalars[s], step=self.step)
            
    def _record_images(self, **images):
        for i in images:
            tf.summary.image(i, images[i], step=self.step)
            
    def _record_hists(self, **hists):
        for h in hists:
            tf.summary.histogram(h, hists[h], step=self.step)
                      
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            