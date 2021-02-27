# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
import os

from patchwork.loaders import dataset, _fixmatch_unlab_dataset
from patchwork.feature._generic import GenericExtractor
from patchwork._util import compute_l2_loss
from patchwork._sample import PROTECTED_COLUMN_NAMES

from sklearn.metrics import roc_auc_score

INPUT_PARAMS = ["imshape", "num_channels", "norm", "batch_size",
                "shuffle", "num_parallel_calls", "sobel", "single_channel"]

DEFAULT_WEAK_AUG = {"flip_left_right":True}
DEFAULT_STRONG_AUG = {'zoom_scale': 0.5, 'jitter': 1.0, 'flip_left_right': True, 
                      'drop_color': 0.25, 'mask': 1.,
                      "gaussian_blur":0.4, "gaussian_noise":0.4,
                      "shear":0.3, "solarize":0.25, "autocontrast":0.25}


def _build_mask(x, tau):
    # helper to generate mask for FixMatch. slightly different from
    # the paper since we're doing sigmoid multilabel learning
    confident_high = x >= tau
    confident_low = x <= (1-tau)
    mask = tf.math.logical_or(confident_high, confident_low)
    return tf.cast(mask, tf.float32)
                
            
def _build_fixmatch_training_step(model, optimizer, lam=0, 
                            tau=0.95, weight_decay=0):
    """
    Generate the training step function
    
    :model: Keras model
    :optimizer: Keras optimizer
    :lam: float; fixmatch loss weight
    :tau: float between 0 and 1; fixmatch threshold
    :weight_decay: float; L2 loss weight
    """
    trainvars = model.trainable_variables
    
    
    def train_step(lab, unlab):
        x,y = lab
        x_unlab_wk, x_unlab_str = unlab
        
        if lam > 0:
            # GENERATE FIXMATCH PSEUDOLABELS
            # make predictions on the weakly-augmented batch
            unlab_preds = model(x_unlab_wk, training=False)
            # round predictions to pseudolabels
            pseudolabels = tf.cast(unlab_preds > 0.5, 
                                           tf.float32)
            # also compute a mask from the predictions,
            # since we only incorporate high-confidence cases,
            # compute a mask that's 1 every place that's close
            # to 1 or 0
            mask = _build_mask(unlab_preds, tau)
        
        with tf.GradientTape() as tape:
            preds = model(x, training=True)
            
            trainloss = tf.reduce_mean(K.binary_crossentropy(y, preds))
            total_loss = trainloss
            
            if weight_decay > 0:
                l2_loss = compute_l2_loss(model)
                total_loss += weight_decay*l2_loss
            else:
                l2_loss = 0
            
            # semi-supervised case- loss function for unlabeled data
            # entropy regularization
            if lam > 0:                
                # MAKE PREDICTIONS FROM STRONG AUGMENTATION
                str_preds = model(x_unlab_str, training=True)
                # let's try keeping track of how accurate these
                # predictions are
                ssl_acc = tf.reduce_mean(tf.cast(
                    tf.cast(str_preds > 0.5, tf.float32)==pseudolabels,
                                                 tf.float32))
                
                crossent_tensor = K.binary_crossentropy(pseudolabels,
                                                        str_preds)
                fixmatch_loss = tf.reduce_mean(mask*crossent_tensor)
                total_loss += lam*fixmatch_loss
            else:
                fixmatch_loss = 0
                ssl_acc = -1
                
        # compute and apply gradients
        gradients = tape.gradient(total_loss, trainvars)
        optimizer.apply_gradients(zip(gradients, trainvars))

        return {"total_loss":total_loss, "supervised_loss":trainloss,
                "fixmatch_loss":fixmatch_loss, "l2_loss":l2_loss,
                "fixmatch_prediction_accuracy":ssl_acc}
    return train_step
            

         
def _fixmatch_dataset(labeled_filepaths, labels, unlabeled_filepaths, 
                      imshape, num_parallel_calls, norm,
                      num_channels, single_channel, batch_size,
                      weak_aug, strong_aug, mu):
    """
    Build the training dataset. We're going to be zipping together two
    datasets:
        -a conventional supervised dataset generating weakly-augmented
        (image, label) pairs
        -a dataset generating unabeled pairs of the same image, one weakly-augmented
        and the other strongly-augmented, for the semisupervised training component
    """
    # dataset for supervised task
    sup_ds = dataset(labeled_filepaths, ys=labels, imshape=imshape, 
                             num_parallel_calls=num_parallel_calls,
                                 norm=norm, num_channels=num_channels,
                                 single_channel=single_channel,
                                 batch_size=batch_size,
                                 augment=weak_aug, shuffle=True)[0]
    
    # dataset for unsupervised task
    unsup_ds = _fixmatch_unlab_dataset(unlabeled_filepaths, weak_aug, strong_aug,
                                       imshape=imshape, num_parallel_calls=num_parallel_calls,
                                       norm=norm, num_channels=num_channels,
                                       single_channel=single_channel, 
                                       batch_size=mu*batch_size)
    # zipped dataset is ((x,y), (x_wk, x_str))
    ds = tf.data.Dataset.zip((sup_ds, unsup_ds))
    ds = ds.prefetch(1)
    return ds



class FixMatchTrainer(GenericExtractor):
    """
    Class for managing semisupervised multlabel training with FixMatch.
    
    The training and validation data should be pandas dataframes each containing
    a "filepath" column (giving location of the image), as well as one column
    for each binary category (with values 0 or 1). Unlike some of the other
    tools in this repo, this trainer assumes no missing labels.
    
    The model should be a Keras model with sigmoid outputs (one per category
    column in your data)
    """
    
    
    def __init__(self, logdir, trainingdata, unlabeled_filepaths, valdata,
                 model, weak_aug=DEFAULT_WEAK_AUG, strong_aug=DEFAULT_STRONG_AUG, 
                 lam=1, tau=0.95, mu=4, passes_per_epoch=1,
                 weight_decay=3e-4,
                 lr=1e-3, lr_decay=0, decay_type="exponential", opt_type="momentum",
                 imshape=(256,256), num_channels=3,
                 norm=255, batch_size=64, num_parallel_calls=None,
                 single_channel=False, notes="",
                 strategy=None):
        """
        :logdir: (string) path to log directory
        :trainingdata: pandas dataframe of training data
        :unlabeled_filepaths:
        :valdata: pandas dataframe of validation data
        :model: Keras model to be trained
        :weak_aug: dictionary of weak augmentation parameters- usually just flipping. This
            will be used for FixMatch pseudolabels as well as supervised training
        :strong_aug: dictionary of strong augmentation parameters, for FixMatch predictions
        :lam: FixMatch lambda parameter; semisupervised loss weight
        :tau: FixMatch threshold parameter
        :mu: FixMatch batch size multiplier
        :passes_per_epoch: for small labeled datasets- run through the data this many times 
            per epoch
        :weight_decay: (float) coefficient for L2-norm loss.
        :lr: learning rate
        :lr_decay: learning rate decay (set to 0 to disable)
        :decay_type: (str) how to decay learning rate; "exponential", "cosine", or "staircase"
        :opt_type: (str) optimizer type
        :imshape: (tuple) image dimensions in H,W
        :num_channels: (int) number of image channels
        :norm: (int or float) normalization constant for images (for rescaling to
               unit interval)
        :batch_size: (int) batch size for training
        :num_parallel_calls: (int) number of threads for loader mapping
        :single_channel: if True, expect a single-channel input image and 
            stack it num_channels times.
        :notes: any experimental notes you want recorded in the config.yml file
        :strategy: if distributing across multiple GPUs, pass a tf.distribute
            Strategy object here
        """
        self.logdir = logdir
        self._weak_aug = weak_aug
        self._strong_aug = strong_aug
        self.strategy = strategy
        self.model = model
        # find the columns of the dataframe that correspond to binary class labels
        self.categories = [c for c in trainingdata.columns if c not in PROTECTED_COLUMN_NAMES]
        self.valdata = valdata
        self._val_labels = valdata[self.categories].values
        self._models = {"full":model}
        self._passes_per_epoch = passes_per_epoch
        
        
        # create optimizer
        self._optimizer = self._build_optimizer(lr, lr_decay, 
                                                decay_type=decay_type,
                                                opt_type=opt_type)
        
        
        # build our training dataset
        self._ds = self._distribute_dataset(
            _fixmatch_dataset(trainingdata["filepath"].values,
                              trainingdata[self.categories].values.astype(np.float32),
                              unlabeled_filepaths,
                              imshape=imshape, num_parallel_calls=num_parallel_calls,
                              norm=norm, num_channels=num_channels, single_channel=single_channel,
                              batch_size=batch_size, weak_aug=weak_aug, strong_aug=strong_aug,
                              mu=mu))
        # and validation dataset
        self._val_ds = dataset(valdata["filepath"].values,
                                imshape=imshape, num_parallel_calls=num_parallel_calls,
                                 norm=norm, num_channels=num_channels,
                                 single_channel=single_channel,
                                 augment=False, shuffle=False)[0]
        
        # build training step
        trainstep = _build_fixmatch_training_step(model, self._optimizer,
                                                  lam=lam, tau=tau,
                                                  weight_decay=weight_decay)
        self._training_step = self._distribute_training_function(trainstep)
        
        self._file_writer = tf.summary.create_file_writer(logdir, flush_millis=10000)
        self._file_writer.set_as_default()
        self.step = 0
        
        self._parse_configs(augment=strong_aug, 
                            lam=lam, tau=tau, mu=mu,
                            weight_decay=weight_decay,
                            lr=lr, lr_decay=lr_decay, decay_type=decay_type, 
                            opt_type=opt_type, imshape=imshape, 
                            num_channels=num_channels, norm=norm, batch_size=batch_size,
                            num_parallel_calls=num_parallel_calls, single_channel=single_channel,
                            notes=notes, 
                            trainer="fixmatch")
        
        


    def _run_training_epoch(self, **kwargs):
        for _ in range(self._passes_per_epoch):
            for lab, unlab in self._ds:
                lossdict = self._training_step(lab, unlab)
                self._record_scalars(**lossdict)
                self._record_scalars(learning_rate=self._get_current_learning_rate())
                self.step += 1
        
            
    def evaluate(self, avpool=True):
        predictions = self._models["full"].predict(self._val_ds)
        num_categories = len(self.categories)
        
        for i in range(num_categories):
            category = self.categories[i]
            preds = predictions[:,i]
            y_true = self._val_labels[:,i]
            
            acc = np.mean(y_true == (preds >= 0.5).astype(int))

            auc = roc_auc_score(y_true, preds)
            self._record_scalars(**{f"val_accuracy_{category}":acc,
                                    f"val_auc_{category}":auc}, metric=True)
            
        # choose the hyperparameters to record
        if not hasattr(self, "_hparams_config"):
            from tensorboard.plugins.hparams import api as hp
            
            metrics = []
            for c in self.categories:
                metrics.append(f"val_accuracy_{c}")
                metrics.append(f"val_auc_{c}")
                
            hparams = {
                    hp.HParam("lam", hp.RealInterval(0., 10000.)):self.config["lam"],
                    hp.HParam("tau", hp.RealInterval(0., 10000.)):self.config["tau"],
                    hp.HParam("mu", hp.RealInterval(0., 10000.)):self.config["mu"],
                    hp.HParam("batch_size", hp.RealInterval(0., 10000.)):self.input_config["batch_size"],
                    hp.HParam("lr", hp.RealInterval(0., 10000.)):self.config["lr"],
                    hp.HParam("lr_decay", hp.RealInterval(0., 10000.)):self.config["lr_decay"],
                    hp.HParam("decay_type", hp.Discrete(["cosine", "exponential", "staircase"])):self.config["decay_type"],
                    hp.HParam("opt_type", hp.Discrete(["sgd", "adam", "momentum"])):self.config["opt_type"],
                    hp.HParam("weight_decay", hp.RealInterval(0., 10000.)):self.config["weight_decay"]
                    }
            
            self._hparams_config = hp.hparams_config(
                        hparams=list(hparams.keys()), 
                        metrics=[hp.Metric(m) for m in metrics])
            # record hyperparameters
            base_dir, run_name = os.path.split(self.logdir)
            if len(run_name) == 0:
                base_dir, run_name = os.path.split(base_dir)
                
                
    def log_model(self, model_name="fixmatch_model"):
        """
        Log the feature extractor to an MLflow server. Assumes you've
        already run track_with_mflow()
        
        Overwriting the generic version of this function because we're not
        building an FCN here.
        """
            
        assert hasattr(self, "_mlflow"), "need to run track_with_mlflow() first"
        from mlflow.keras import log_model
        log_model(self._models["full"], model_name)
        
  
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            