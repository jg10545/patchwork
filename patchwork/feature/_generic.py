"""

            _generic.py

I feel like I'm rewriting a lot of the same code- so let's define a master
class holding all the stuff we want to reuse between feature extractors.

"""
import os
import yaml
import numpy as np

import tensorflow as tf
from tqdm import tqdm
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from tensorboard.plugins.hparams import api as hp

from patchwork.loaders import _get_features, _get_rotation_features
from patchwork.viz._projector import save_embeddings
from patchwork.viz._kernel import _make_kernel_sprites
from patchwork._mlflow import _set_up_mlflow_tracking


INPUT_PARAMS = ["imshape", "num_channels", "norm", "batch_size",
                "shuffle", "num_parallel_calls", "single_channel",
                "global_batch_size"]


def _run_this_epoch(flag, e): 
    if isinstance(flag, bool):
        return flag
    else:
        if e > 0:
            return flag%e ==0 
        else:
            return False


def linear_classification_test(fcn, downstream_labels, avpool=False, rotation_task=False,
                               **input_config):
    """
    Train a linear classifier on a fully-convolutional network
    and return out-of-sample results.
    
    :fcn: Keras fully-convolutional network
    :downstream_labels: dictionary mapping image file paths to labels, or a 
        dataset returning image/label pairs
    :avpool: average-pool feature tensors before fitting linear model. if False, flatten instead.
    :rotation_task: if True, use an unlabeled list of filepaths and measure performance on an
        image-rotation task (see "Evaluating Self-Supervised Pretraining Without Using Labels")
    :input_config: kwargs for patchwork.loaders.dataset()
    
    Returns
    :acc: float; test accuracy
    :cm: 2D numpy array; confusion matrix
    """
    # load features into memory
    if rotation_task:
        features, labels = _get_rotation_features(fcn, 
                                            downstream_labels, **input_config)     
        
    else:
        features, labels = _get_features(fcn, 
                                            downstream_labels, **input_config)     
    # build a deterministic train/test split
    split = np.array([(i%3 == 0) for i in range(features.shape[0])])
    trainvecs = features[~split]
    testvecs = features[split]
    
    # rescale train and test
    scaler = StandardScaler().fit(trainvecs)
    trainvecs = scaler.transform(trainvecs)
    testvecs = scaler.transform(testvecs)
    # train a multinomial linear classifier
    logreg = SVC(kernel="linear")
    logreg.fit(trainvecs, labels[~split])
    # make predictions on test set
    preds = logreg.predict(testvecs)
    # compute metrics and return
    acc = accuracy_score(labels[split], preds)
    cm = confusion_matrix(labels[split], preds)
    return acc, cm



def build_optimizer(lr, lr_decay=0, opt_type="adam", decay_type="exponential"):
    """
    Macro to reduce some duplicative code for building optimizers
    for trainers
    
    :decay_type: exponential or cosine
    """
    if lr_decay > 0:
        if decay_type == "exponential":
            lr = tf.keras.optimizers.schedules.ExponentialDecay(lr, 
                                        decay_steps=lr_decay, decay_rate=0.5,
                                        staircase=False)
        elif decay_type == "staircase":
            lr = tf.keras.optimizers.schedules.ExponentialDecay(lr, 
                                        decay_steps=lr_decay, decay_rate=0.5,
                                        staircase=True)
        elif decay_type == "cosine":
            lr = tf.keras.experimental.CosineDecayRestarts(lr, lr_decay,
                                                           t_mul=2., m_mul=1.,
                                                           alpha=0.)
        else:
            assert False, "don't recognize this decay type"
    if opt_type == "adam":
        return tf.keras.optimizers.Adam(lr)
    elif opt_type == "momentum":
        return tf.keras.optimizers.SGD(lr, momentum=0.9)
    else:
        assert False, "dont know what to do with {}".format(opt_type)


class EmptyContextManager():
    def __init__(self):
        pass
    def __enter__(self):
        pass
    def __exit__(self, *args):
        pass



class GenericExtractor(object):
    """
    Place to store common code for different feature extractor methods. Don't 
    actually use this to do anything.
    
    To subclass this, replace:
        __init__
        _build_default_model
        _run_training_epoch
        evaluate
    """
    modelname = "GenericExtractor"
    
    
    def __init__(self, logdir=None, trainingdata=[], fcn=None, augment=False, 
                 extractor_param=None, imshape=(256,256), num_channels=3,
                 norm=255, batch_size=64, shuffle=True, num_parallel_calls=None,
                 sobel=False, single_channel=False, strategy=None):
        """
        :logdir: (string) path to log directory
        :trainingdata: (list or tf Dataset) list of paths to training images, or
            dataset to use for training loop
        :fcn: (keras Model) fully-convolutional network to train as feature extractor
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
        """
        self.logdir = logdir
        self.strategy = strategy
        
        if fcn is None:
            fcn = self._build_default_model()
        self.fcn = fcn
        self._models = {"fcn":fcn}
        
        if logdir is not None:
            self._file_writer = tf.summary.create_file_writer(logdir, flush_millis=10000)
            self._file_writer.set_as_default()
        self.step = 0
        
        
        
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
                
        if "notes" in kwargs:
            self._notes = kwargs["notes"]
        else:
            self._notes = None
                
        config_path = os.path.join(self.logdir, "config.yml")
        config_dict = {"model":self.config, "input":self.input_config, 
                       "augment":self.augment_config}
        yaml.dump(config_dict, open(config_path, "w"), default_flow_style=False)
        
        
    def _build_default_model(self, **kwargs):
        # REPLACE THIS WHEN SUBCLASSING
        return True
    
    def _run_training_epoch(self, **kwargs):
        # REPLACE THIS WHEN SUBCLASSING
        return True
    
    def fit(self, epochs=1, save=True, evaluate=True, save_projections=False,
            visualize_kernels=False):
        """
        Train the feature extractor. All kwargs after "epochs" can either be
        Boolean (whether to run after every epoch) or an integer (run after
        every N epochs)
        
        :epochs: number of epochs to train for
        :save: save every model in the trainer
        :evaluate: run eval metrics
        :save_projections: record projections from labeldict
        :visualize_kernels: record a visualization of the first convolutional
            layer's kernels
        """
        for e in tqdm(range(epochs)):
            self._run_training_epoch()
            
            if _run_this_epoch(save, e):
                self.save()
            if _run_this_epoch(evaluate, e):
                self.evaluate()
            if _run_this_epoch(save_projections, e):
                self.save_projections()
            if _run_this_epoch(visualize_kernels,e):
                self.visualize_kernels()
    
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
        # REPLACE THIS WHEN SUBCLASSING
        return True
            
    def _record_scalars(self, metric=False, **scalars):
        for s in scalars:
            tf.summary.scalar(s, scalars[s], step=self.step)
            
            if metric:
                if hasattr(self, "_mlflow"):
                    self._log_metrics(scalars, step=self.step)
            
    def _record_images(self, **images):
        for i in images:
            tf.summary.image(i, images[i], step=self.step)
            
    def _record_hists(self, **hists):
        for h in hists:
            tf.summary.histogram(h, hists[h], step=self.step)
            
    def _linear_classification_test(self, params=None, 
                                    metrics=["linear_classification_accuracy"]):
         acc, conf_mat = linear_classification_test(self.fcn, 
                                    self._downstream_labels, 
                                    **self.input_config)
         
         conf_mat = np.expand_dims(np.expand_dims(conf_mat, 0), -1)/conf_mat.max()
         self._record_scalars(linear_classification_accuracy=acc, metric=True)
         self._record_images(linear_classification_confusion_matrix=conf_mat)
         # if the model passed hyperparameters to record for the
         # tensorboard hparams interface:
         if params is not None:
             # first time- set up hparam config
             if not hasattr(self, "_hparams_config"):
                metrics = [hp.Metric(m) for m in metrics]
                self._hparams_config = hp.hparams_config(
                        hparams=list(params.keys()), 
                        metrics=metrics)
                
                # record hyperparamters
                base_dir, run_name = os.path.split(self.logdir)
                if len(run_name) == 0:
                    base_dir, run_name = os.path.split(base_dir)
                hp.hparams(params, trial_id=run_name)
                
    def rotation_classification_test(self, testdata=None, avpool=False):
        """
        Test the feature extractor's performance on a rotation-prediction
        task. See "Evaluating Self-Supervised Pretraining Without Using 
        Labels" by Reed et al for why you'd want to do this!
        
        https://arxiv.org/abs/2009.07724
        
        :testdata: a list of filepaths, or a tf Dataset that loads and
            returns single images
        :avpool: if True, use average pooling instead of flattening.
        """
        if testdata is None:
            if hasattr(self, "_testdata"):
                testdata = self._testdata
            else:
                assert False, "need images to sample from"
        
        acc, conf_mat = linear_classification_test(self.fcn, 
                                    testdata, avpool=avpool,
                                    rotation_task=True,
                                    **self.input_config)
        self._record_scalars(rotation_classification_accuracy=acc, metric=True)
                
    def _build_optimizer(self, lr, lr_decay=0, opt_type="adam", decay_type="exponential"):
        # macro for creating the Keras optimizer
        with self.scope():
            opt = build_optimizer(lr, lr_decay,opt_type, decay_type)
        return opt
    

        
            
    def _born_again_loss_function(self):
        """
        Generate a function that computes the Born-Again Network loss
        if a "teacher" model is in the model dictionary
        """
        if "teacher" in self._models:
            teacher = self._models["teacher"]
            student = self._models["full"]
            assert len(teacher.outputs) == len(student.outputs), "number of outputs don't match"
            def _ban_loss(student_outputs, x):
                """
                Computes Kullback-Leibler divergence between student
                and teacher model outputs
                
                Note that right now this function assumes multiple-output
                models (hence the correction to make a single-output model
                        a list of length 1). i should make this more robust.
                
                :student_outputs: model outputs from the model being trained 
                        (since we're probably already computing those elsewhere 
                        in the training step)
                :x: current training batch
                """
                teacher_outputs = self._models["teacher"](x)
                if isinstance(teacher_outputs, tf.Tensor):
                    teacher_outputs = [teacher_outputs]
                loss = 0
                for s,t in zip(student_outputs, teacher_outputs):
                    loss += tf.reduce_mean(tf.keras.losses.KLD(t,s))
                return loss
            return _ban_loss
        else:
            return None
        
    def scope(self):
        """
        
        """
        if hasattr(self, "strategy") and (self.strategy is not None):
            return self.strategy.scope()
        else:
            return EmptyContextManager()
        
    def _distribute_training_function(self, step_fn):
        """
        Pass a tensorflow function and distribute if necessary
        """
        if self.strategy is None:
            @tf.function
            def training_step(x,y):
                return step_fn(x,y)
        else:
            @tf.function
            def training_step(x,y):
                per_example_losses = self.strategy.experimental_run_v2(
                                        step_fn, args=(x,y))

                lossdict = {k:self.strategy.reduce(
                    tf.distribute.ReduceOp.MEAN, 
                    per_example_losses[k], axis=None)
                    for k in per_example_losses}

                return lossdict
        return training_step
        
    def _distribute_dataset(self, ds):
        """
        Pass a tensorflow dataset and distribute if necessary
        """
        if self.strategy is None:
            return ds
        else:
            return self.strategy.experimental_distribute_dataset(ds)
        
    def _get_current_learning_rate(self):
        # return the current value of the learning rate
        # CONSTANT LR CASE
        if isinstance(self._optimizer.lr, tf.Variable) or isinstance(self._optimizer.lr, tf.Tensor):
            return self._optimizer.lr
        # LR SCHEDULE CASE
        else:
            return self._optimizer.lr(self.step)
                
    def save_projections(self, proj_dim=0, sprite_size=50):
        """
        Use Tensorboard's projector to visualize the embeddings of images
        in the downstream_labels dictionary or dataset. It does all this in 
        memory, so probably not a great idea to call this if you have a
        million labels.
        
        Each image is run through the FCN and then flattened.
        
        :proj_dim: Use PCA to reduce the dimension before saving. 0 
                to disable
        :sprite_size: size of each sprite, in pixels. For now this 
                function assumes the patches are square.
        """
        labels = self._downstream_labels
        assert isinstance(labels, dict) & (len(labels)>0), "dont you need some labels?"
        save_embeddings(self._models["fcn"], labels, self.logdir,
                        proj_dim, sprite_size, **self.input_config)
        
    def load_weights(self, logdir):
        """
        Update model weights from a previously trained model
        """
        for k in self._models:
            savedloc = os.path.join(logdir, k+".h5")
            self._models[k].load_weights(savedloc)

    def visualize_kernels(self):
        """
        Save a visualization to TensorBoard of all the kernels in the first
        convolutional layer
        """
        kernels = np.expand_dims(_make_kernel_sprites(self._models["fcn"]),0)
        self._record_images(first_convolution_filters=kernels)
        
    def track_with_mlflow(self, tracking_uri, experiment_name, run_name=None):
        """
        Connect to an MLflow server to log parameters and metrics
        
        :tracking_uri: string; Address of local or remote tracking server.
        :experiment_name: string; name of experiment to log to. if experiment 
                doesn't exist, creates one with this name
        :run_name: string; name of run to log to. if run doesn't exist, 
            one will be created
        """
        # set up a connection to the server, an experiment, and the run
        import mlflow
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        run = mlflow.start_run(run_name=run_name)
        
        self._mlflow = {"run":run}
        # log all our parameters for the model, input configuration, and
        # data augmentation
        mlflow.log_params({"model_"+k:self.config[k] for k in self.config})
        mlflow.log_params({"input_"+k:self.input_config[k] for k in 
                           self.input_config})
        if self.augment_config is not False:
            mlflow.log_params({"augment_"+k:self.augment_config[k] for k in 
                           self.augment_config})
        if self._notes is not None:
            mlflow.set_tag("mlflow.note.content", self._notes)
            
        self._log_metrics = mlflow.log_metrics

        
    def log_model(self, model_name=None):
        """
        Log the feature extractor to an MLflow server. Assumes you've
        already run track_with_mflow()
        """
        if model_name is None:
            model_name = self.modelname + "_FCN"
            
        assert hasattr(self, "_mlflow"), "need to run track_with_mlflow() first"
        from mlflow.keras import log_model
        log_model(self._models["fcn"], model_name)
        
    def __del__(self):
        if hasattr(self, "_mlflow"):
            import mlflow
            mlflow.end_run()
        
