"""

            _generic.py

I feel like I'm rewriting a lot of the same code- so let's define a master
class holding all the stuff we want to reuse between feature extractors.

"""
import os
import yaml
import numpy as np

import tensorflow as tf
from tensorboard.plugins.hparams import api as hp
from tqdm import tqdm
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from tensorboard.plugins.hparams import api as hp

from patchwork.loaders import _get_features, _get_rotation_features
from patchwork.viz._projector import save_embeddings
from patchwork.viz._kernel import _make_kernel_sprites
from patchwork.viz._feature import _build_query_image
from patchwork._util import build_optimizer


INPUT_PARAMS = ["imshape", "num_channels", "norm", "batch_size",
                "shuffle", "num_parallel_calls", "single_channel",
                "global_batch_size"]

_TENSORBOARD_DESCRIPTIONS = {
    "loss":"Total training loss",
    "nce_batch_acc":"Accuracy of the contrastive training batch. *Hard Negative Mixing for Contrastive Learning* by Kalantidis *et al* uses this plot to gain some insight into differences between variations on MoCo.",
    "alignment":"Alignment measure from Wang and Isola's *Understanding Contrastive Representation Learning through Alignment and Uniformity on the Hypersphere*. Measures how well-aligned positive pairs are (higher is better). Computes on test files.",
    "uniformity":"Uniformity measure from Wang and Isola's *Understanding Contrastive Representation Learning through Alignment and Uniformity on the Hypersphere*. Measures how well spread feature vectors are over the unit hypersphere (lower is better). Computes on test files.",
    "l2_loss":"Total squared magnitude of training weights for L2 loss computation. This is the value **before** rescaling by your weight decay parameter.",
    "linear_classification_accuracy":"Downstream task test accuracy. Feature vectors for labeled images are average-pooled and used to train a multinomial regression model. Labeled points are deterministically split into 2/3 train, 1/3 test.",
    "first_convolution_filters":"Kernels from the first convolutional layer of the feature extractor.",
    
}


def _run_this_epoch(flag, e): 
    if isinstance(flag, bool):
        return flag
    else:
        if e > 0:
            return e%flag == 0  
        else:
            return False


SPECIAL_HPARAMS = {
    "decay_type":["cosine", "exponential", "staircase"],
    "opt_type":["adam", "momentum"]
    }

def _configure_hparams(logdir, dicts, 
                       metrics=["linear_classification_accuracy",
                                "alignment", "uniformity"]):
    """
    Set up the tensorboard hyperparameter interface
    
    :logdir: string; path to log directory
    :dicts: list of dictionaries containing hyperparameter values
    :metrics: list of strings; metric names
    """
    metrics = [hp.Metric(m) for m in metrics]
    params = {}
    # for each parameter dictionary
    for d in dicts:
        # for each parameter:
        for k in d:
            # is it a categorical?
            if k in SPECIAL_HPARAMS:
                params[hp.HParam(k, hp.Discrete(SPECIAL_HPARAMS[k]))] = d[k]
            elif isinstance(d[k], bool):
                params[hp.HParam(k, hp.Discrete([True, False]))] = d[k]
            elif isinstance(d[k], int):
                params[hp.HParam(k, hp.IntInterval(1, 1000000))] = d[k]
            elif isinstance(d[k], float):
                params[hp.HParam(k, hp.RealInterval(0., 10000000.))] = d[k]
    #
    hparams_config = hp.hparams_config(
                        hparams=list(params.keys()), 
                        metrics=metrics)
                
    # get a name for the run
    base_dir, run_name = os.path.split(logdir)
    if len(run_name) == 0:
        base_dir, run_name = os.path.split(base_dir)
    # record hyperparamers
    hp.hparams(params, trial_id=run_name)
    

def linear_classification_test(fcn, downstream_labels, avpool=True, rotation_task=False,
                               query_fig=False, **input_config):
    """
    Train a linear classifier on a fully-convolutional network
    and return out-of-sample results.
    
    :fcn: Keras fully-convolutional network
    :downstream_labels: dictionary mapping image file paths to labels, or a 
        dataset returning image/label pairs
    :avpool: average-pool feature tensors before fitting linear model. if False, flatten instead.
    :rotation_task: if True, use an unlabeled list of filepaths and measure performance on an
        image-rotation task (see "Evaluating Self-Supervised Pretraining Without Using Labels")
    :query_fig: set to an integer to return data for a figure showing images with similar vectors
    :input_config: kwargs for patchwork.loaders.dataset()
    
    Returns
    :acc: float; test accuracy
    :cm: 2D numpy array; confusion matrix
    :fig: 4D numpy array; only returned if query_fig > 0
    """
    # load features into memory
    if rotation_task:
        features, labels = _get_rotation_features(fcn, downstream_labels, 
                                                  avpool=avpool, **input_config)     
        
    else:
        features, labels = _get_features(fcn, downstream_labels, 
                                            avpool=avpool, **input_config)     
    # build a deterministic train/test split
    split = np.array([(i%3 == 0) for i in range(features.shape[0])])
    trainvecs = features[~split]
    testvecs = features[split]
    
    # rescale train and test
    scaler = StandardScaler().fit(trainvecs)
    trainvecs = scaler.transform(trainvecs)
    testvecs = scaler.transform(testvecs)
    # train a multinomial linear classifier
    logreg = SGDClassifier(loss="log", max_iter=1000, n_jobs=-1, 
                           learning_rate="adaptive", eta0=1e-2)
    logreg.fit(trainvecs, labels[~split])
    # make predictions on test set
    preds = logreg.predict(testvecs)
    # compute metrics and return
    acc = accuracy_score(labels[split], preds)
    cm = confusion_matrix(labels[split], preds)
    
    if query_fig:
        fig = _build_query_image(features, labels, downstream_labels, query_fig)
        return acc, cm, fig
    return acc, cm





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
    _description = _TENSORBOARD_DESCRIPTIONS
    
    
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
        
        
        
    def _parse_configs(self, metrics=["linear_classification_accuracy",
                                      "alignment", "uniformity"], **kwargs):
        """
        Organize input parameters and save to a YAML file so you can
        find them later.
        """
        self.config = {}
        self.input_config = {}
        self.augment_config = False
        # separate out input params, augmentation params, and model-specific
        # params
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
                
        # dump configuration to a YAML file
        config_path = os.path.join(self.logdir, "config.yml")
        config_dict = {"model":self.config, "input":self.input_config, 
                       "augment":self.augment_config}
        yaml.dump(config_dict, open(config_path, "w"), default_flow_style=False)
        
        # save tensorboard hyperparameters
        # the only input parameter relevant to most hyperparameter tuning
        # questions is the batch size.
        dicts = [self.config, {"batch_size":self.input_config["batch_size"]},
                 self.augment_config]
        _configure_hparams(self.logdir, dicts, metrics)
        
    def _build_default_model(self, **kwargs):
        # REPLACE THIS WHEN SUBCLASSING
        return True
    
    def _run_training_epoch(self, **kwargs):
        # REPLACE THIS WHEN SUBCLASSING
        return True
    
    def fit(self, epochs=1, avpool=True, save=True, evaluate=True, 
            save_projections=False,
            visualize_kernels=False, log_fcn=False, query_fig=False):
        """
        Train the feature extractor. All kwargs after "avpool" can either be
        Boolean (whether to run after every epoch) or an integer (run after
        every N epochs)
        
        :epochs: number of epochs to train for
        :avpool: if True, use average-pooled features for linear classifications
            during the evaluation step. if False, flatten the features.
        :save: save every model in the trainer
        :evaluate: run eval metrics
        :save_projections: record projections from labeldict
        :visualize_kernels: record a visualization of the first convolutional
            layer's kernels
        :log_fcn: log a copy of the FCN with the epoch number so it won't be 
            overwritten
        :query_fig: set to an integer to build a figure showing one example image
            per class, along with <query_fig> images with closest feature vectors
        
        """
        for e in tqdm(range(epochs)):
            self._run_training_epoch()
            
            if _run_this_epoch(save, e):
                self.save()
            if _run_this_epoch(evaluate, e):
                self.evaluate(avpool=avpool, query_fig=query_fig)
            if _run_this_epoch(save_projections, e):
                self.save_projections()
            if _run_this_epoch(visualize_kernels,e):
                self.visualize_kernels()
            if _run_this_epoch(log_fcn, e):
                self._log_fcn(e)
    
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
            
    def _log_fcn(self, e):
        """
        Write the FCN to disk with 'e' appended
        """
        path = os.path.join(self.logdir, f"fcn_{e}.h5")
        self._models["fcn"].save(path, save_format="h5")
            
    def evaluate(self):
        # REPLACE THIS WHEN SUBCLASSING
        return True
            
    def _record_scalars(self, metric=False, **scalars):
        for s in scalars:
            desc = self._description.get(s, None)
            tf.summary.scalar(s, scalars[s], step=self.step, description=desc)
            
            if metric:
                if hasattr(self, "_mlflow"):
                    self._log_metrics(scalars, step=self.step)
            
    def _record_images(self, **images):
        for i in images:
            desc = self._description.get(i, None)
            tf.summary.image(i, images[i], step=self.step,
                             description=desc)
            
    def _record_hists(self, **hists):
        for h in hists:
            desc = self._description.get(h, None)
            tf.summary.histogram(h, hists[h], step=self.step,
                                 description=desc)
            
    def _linear_classification_test(self, avpool=True, query_fig=False):
        
         results = linear_classification_test(self.fcn, 
                                    self._downstream_labels, 
                                    avpool=avpool, query_fig=query_fig,
                                    **self.input_config)
         if query_fig:
             acc, conf_mat, fig = results
             self._record_images(closest_feature_vectors=fig)
         else:
             acc, conf_mat =results
         
         conf_mat = np.expand_dims(np.expand_dims(conf_mat, 0), -1)/conf_mat.max()
         self._record_scalars(linear_classification_accuracy=acc, metric=True)
         # commenting out the confusion matrix record- I don't think I've ever found
         # this actually useful
         #self._record_images(linear_classification_confusion_matrix=conf_mat)
  
                
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
                per_example_losses = self.strategy.run(step_fn, args=(x,y))

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
        
