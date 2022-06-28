import numpy as np
import pandas as pd
import panel as pn
import tensorflow as tf
import logging
import os
from tqdm import tqdm



from patchwork._labeler import Labeler
from patchwork._modelpicker import ModelPicker
from patchwork._trainmanager import TrainManager, _auc_and_acc
from patchwork._sample import stratified_sample, find_unlabeled, find_excluded_indices
from patchwork._sample import _build_in_memory_dataset, find_labeled_indices
from patchwork._sample import PROTECTED_COLUMN_NAMES, stratified_subset_sample
from patchwork._training_functions import build_training_function
from patchwork.loaders import dataset, _fixmatch_unlab_dataset
from patchwork._losses import entropy_loss, masked_binary_crossentropy
from patchwork._util import _load_img, build_optimizer
from patchwork._badge import KPlusPlusSampler, _build_output_gradient_function#_v1
from patchwork._labelprop import _get_weighted_adjacency_matrix, _propagate_labels
from patchwork._labelprop import LabelPropagator
from patchwork._error_visualizer import ErrorVisualizer

EPSILON = 1e-5


DEFAULT_FIXMATCH_AUGMENT={"zoom_scale":0.25, "jitter":1.,
                          "flip_left_right":True, "drop_color":0.2,
                          "mask":0.2}


class GUI(object):
    
    def __init__(self, df, feature_vecs=None, feature_extractor=None,
                 classes=[],
                 imshape=(256,256), num_channels=3, norm=255,
                 num_parallel_calls=2, logdir=None, aug=True, 
                 fixmatch_aug=DEFAULT_FIXMATCH_AUGMENT, dim=4,
                 tracking_uri=None, experiment_name=None, strategy=None,
                 diversity_sampling_proj_dim=None, default_prediction_batch_size=64,
                 labelprop_tab=False, errorviz_tab=False):
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
        :logdir: path to directory to save labels and models
        :aug: dict of augmentation parameters. Only matters if you're
            using a feature extractor instead of static features.
        :fixmatch_aug: strong augmentation parameters for semisupervised 
            learning using the FixMatch algorithm
        :dim: grid dimension for labeler- show a (dim x dim) square of images
        :tracking_uri: string; URI of MLFlow tracking server
        :experiment_name: string; name for MLFlow experiment to track
        :strategy: tensorflow strategy object for distributing across GPUs
        :diversity_sampling_proj_dim: set to a positive integer to enable diversity
            sampling. average-pooled feature vectors will be projected to this 
            dimension.
        :default_prediction_batch_size: batch size to use for initial feature generation
            for the diversity sampler and error visualization
        :labelprop_tab: whether to include a GUI tab for label propagation
        :errorviz_tab: whether to include a GUI tab for visualizing features and 
            predictions on validation data
        """
        self.fine_tuning_model = None
        self.df = df.copy()
        self.feature_vecs = feature_vecs
        self.feature_extractor = feature_extractor
        self._aug = aug
        self._badge_sampler = None
        self._fixmatch_aug = fixmatch_aug
        self._default_prediction_batch_size = default_prediction_batch_size
        
        if strategy is None:
            strategy = tf.distribute.get_strategy()
        self._strategy = strategy


        self._imshape = imshape
        self._norm = norm
        self._num_channels = num_channels
        self._num_parallel_calls = num_parallel_calls
        self._semi_supervised = False
        self._domain_adapt = False
        self._num_domains = 0
        self._logdir = logdir
        self.models = {"feature_extractor":feature_extractor, 
                       "feature_extractor_backup":feature_extractor, 
                       "teacher_fine_tuning":None,
                       "teacher_output":None}
        # place to hide hyperparameter info for models
        self._model_params = {}
        self.params = {"entropy_reg_weight":0, "mean_teacher_alpha":0}
        
        
        if "exclude" not in df.columns:
            df["exclude"] = False
            
        for c in classes:
            if c not in df.columns:
                df[c] = np.nan
        self.classes = [x for x in df.columns if x not in PROTECTED_COLUMN_NAMES]
        # initialize dataframe of predictions
        self.pred_df = pd.DataFrame(
                {c:np.random.uniform(0,1,len(df)) for c in self.classes},
                index=df.index)
        

        if diversity_sampling_proj_dim is not None:
            logging.info("computing projected embeddings for diversity sampling")
            self.build_projected_embeddings(default_prediction_batch_size,
                                            diversity_sampling_proj_dim)
        else:
            self._diversity_sampler = None
        
        # BUILD THE GUI
        # initialize Labeler object
        self.labeler = Labeler(self.classes, self.df, self.pred_df, self._load_img,
                               self, dim=dim, logdir=logdir)
        # initialize model picker
        if self.feature_vecs is not None:
            self._feature_shape = self.feature_vecs.shape[1:]
            
        else:
            test_tensor = tf.zeros((1, imshape[0], imshape[1], num_channels),
                                   dtype=tf.float32)
            self._feature_shape = feature_extractor(test_tensor).shape[1:]
            

        self.modelpicker = ModelPicker(len(self.classes),
                                       self._feature_shape, self,
                                       feature_extractor=feature_extractor)
        # make a train manager- pass this object to it
        self.trainmanager = TrainManager(self)
        # make a label propagator
        if labelprop_tab:
            self.labelpropagator = LabelPropagator(self)
        if errorviz_tab:
            self.errorvisualizer = ErrorVisualizer(self)
        # default optimizer
        self._opt = tf.keras.optimizers.Adam(1e-3)


        
        if tracking_uri is not None:
            self._configure_mlflow(tracking_uri, experiment_name)
        
        
        
    def _configure_mlflow(self, tracking_uri, experiment_name):
        """
        Connect to an MLflow server and create or set up an experiment
        """
        import mlflow, mlflow.keras
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        self._mlflow = mlflow
        
    def _mlflow_track_run(self):
        if hasattr(self, "_mlflow"):
            # end the previous run, if there was one
            self._mlflow.end_run()
            df = self.df
            pred = self.pred_df
            # aggregate parameters
            params = {}
            for k in self._model_params["training"]:
                params[k] = self._model_params["training"][k]
            for k in self._model_params["fine_tuning"]:
                if k != "num_params":
                    params[k] = self._model_params["fine_tuning"][k]
            for k in self._model_params["output"]:
                if k != "num_params":
                    params[k] = self._model_params["output"][k]
            for k in self._model_params["semisup"]:
                params["semisup_"+k] = self._model_params["semisup"][k]
            params["num_params"] = self._model_params["fine_tuning"]["num_params"] + \
                                    self._model_params["output"]["num_params"]
                                    
            params["weight_decay"] = self.modelpicker._weight_decay.value
            num_validation_points = df["validation"].sum()
            params["num_validation_points"] = num_validation_points
            num_training = len(df) - find_unlabeled(df).sum() - num_validation_points
            params["num_training_points"] = num_training
            # start a run
            self._mlflow.start_run()
            # log model params and training params
            self._mlflow.log_params(params)
            # compute validation AUC as a metric to log
            for c in self.classes:
                pos_labeled = pred[c][(df[c] == 1)&(df["validation"] == True)].values
                neg_labeled = pred[c][(df[c] == 0)&(df["validation"] == True)].values
                val_auc, val_acc = _auc_and_acc(pos_labeled, neg_labeled, rnd=8)
                self._mlflow.log_metric(f"val_auc_{c}", val_auc, step=0)
                self._mlflow.log_metric(f"val_acc_{c}", val_acc, step=0)
                
    def log_model(self):
        """
        Log the current model to an MLflow server, if one is connected
        """
        if hasattr(self, "_mlflow"):
            self._mlflow.keras.log_model(self.models["full"],
                                         "activelearner_full_model")
        else:
            print("MLflow connection not set up")
            
        
    def _update_unlabeled(self):
        """
        update our array keeping track of unlabeled images
        """
        self.unlabeled_indices = np.arange(self.N)[np.isnan(self.labels)]
      
        
    def panel(self):
        """
        
        """
        tabs = [("Annotate", self.labeler.panel()),
                       ("Model", self.modelpicker.panel()), 
                       ("Train", self.trainmanager.panel())]
        if hasattr(self, "labelpropagator"):
            tabs.append(("Label Propagation", self.labelpropagator.panel()))
        if hasattr(self, "errorvisualizer"):
            tabs.append(("Error Visualizer", self.errorvisualizer.panel()))
        return pn.Tabs(*tabs)
    
    def serve(self, **kwargs):
        """
        wrapper for panel.serve()
        """
        p = self.panel()
        pn.serve(p, title="patchwork labeling adventure", **kwargs)
        
    
    def _fixmatch_unlab_dataset(self, batch_size, mu):
        """
        dataset for unlabeled part of the fixmatch algorithm
        """
        # train on anything not specifically labeled validation
        all_filepaths = self.df.filepath[~self.df.validation].values
        # in the FixMatch paper they use a larger batch size for
        # unlabeled data
        qN = batch_size*mu
        # Make a tensorflow dataset that will return batches of
        # (weakly augmented image, strongly augmented image) pairs
        unlab_ds = _fixmatch_unlab_dataset(all_filepaths, 
                                         self._aug,
                                         self._fixmatch_aug,
                                         imshape=self._imshape,
                                         norm=self._norm,
                                         num_channels=self._num_channels,
                                         num_parallel_calls=self._num_parallel_calls,
                                         batch_size=qN)
        return unlab_ds
    
    def _domain_adaptation_dataset(self, batch_size, mu):
        """
        dataset for generating (image, domain) pairs for labeled/unlabeled
        data for domain generalization algorithms
        """
        bs = int(batch_size*mu)
        filepaths, labels = stratified_subset_sample(self.df, bs)
        self._num_domains = len([s for s in self.ds.subset.unique() if isinstance(s, str)])
        return  dataset(filepaths, labels, imshape=self._imshape, 
                       num_channels=self._num_channels,
                       num_parallel_calls=self._num_parallel_calls, 
                       batch_size=bs,
                       augment=self._aug)[0]
        
    
    def _training_dataset(self, batch_size=32, num_samples=None,
                          sampling="class", indexlist=None):
        """
        Build a single-epoch training set.
        
        Supervised case: returns tf.data.Dataset object with
            structure (x,y)
            
        Semi-supervised case: returns tf.data.Dataset object
            with structure ((x,y), x_unlab)
        """
        if num_samples is None:
            num_samples = len(self.df)
        # LIVE FEATURE EXTRACTOR CASE
        if self.feature_vecs is None:
            files, ys = stratified_sample(self.df, num_samples, sampling=sampling,
                                          indexlist=indexlist)
            # (x,y) dataset
            ds = dataset(files, ys, imshape=self._imshape, 
                       num_channels=self._num_channels,
                       num_parallel_calls=self._num_parallel_calls, 
                       batch_size=batch_size,
                       augment=self._aug)[0]
            
            # include unlabeled data as well if 
            # we're doing semisupervised learning
            if self._semi_supervised:
                unlab_ds = self._fixmatch_unlab_dataset(batch_size, 
                                                        self._model_params["semisup"]["batch_size_multiplier_mu"])
                # stitch together the labeled and unlabeled datasets
                ds = tf.data.Dataset.zip((ds, unlab_ds))
            # or include unlabeled data if we're doing domain adaptation
            elif self._domain_adapt:
                unlab_ds = self._domain_adaptation_dataset(batch_size, 
                                                        self._model_params["semisup"]["batch_size_multiplier_mu"])
                # stitch together the labeled and unlabeled datasets
                ds = tf.data.Dataset.zip((ds, unlab_ds))
                
            

        # PRE-EXTRACTED FEATURE CASE
        else:
            inds, ys = stratified_sample(self.df, num_samples, return_indices=True)
            if self._semi_supervised:
                unlabeled_indices = np.arange(len(self.df))[find_unlabeled(self.df)]
            else:
                unlabeled_indices = None
                
            ds = _build_in_memory_dataset(self.feature_vecs, 
                                          inds, ys, batch_size=batch_size,
                                          unlabeled_indices=unlabeled_indices)
        return self._strategy.experimental_distribute_dataset(ds)

    def _pred_dataset(self, batch_size=None):
        """
        Build a dataset for predictions
        """
        if batch_size is None:
            batch_size = self._default_prediction_batch_size
        num_steps = int(np.ceil(len(self.df)/batch_size))
        if self.feature_vecs is None:
            files = self.df["filepath"].values
            ds, num_steps = dataset(files, imshape=self._imshape, 
                       num_channels=self._num_channels,
                       num_parallel_calls=self._num_parallel_calls, 
                       batch_size=batch_size, shuffle=False,
                       augment=False)
        # PRE-EXTRACTED FEATURE CASE
        else:
            ds = tf.data.Dataset.from_tensor_slices(self.feature_vecs
                                                      ).batch(batch_size)#, num_steps
        return self._strategy.experimental_distribute_dataset(ds), num_steps
            
        
        
    def _val_dataset(self, batch_size=None):
        """
        Build a dataset of just validation examples
        """
        if batch_size is None:
            batch_size = self._default_prediction_batch_size
        val_df = self.df[self.df.validation]
        ys = val_df[self.classes].values.copy()
        ys[np.isnan(ys)] = -1
        
        if self.feature_vecs is None:
            files = val_df["filepath"].values
            ds, _ =  dataset(files, ys=ys, imshape=self._imshape, 
                       num_channels=self._num_channels,
                       num_parallel_calls=self._num_parallel_calls, 
                       batch_size=batch_size, shuffle=False,
                       augment=False)
        # PRE-EXTRACTED FEATURE CASE
        else:
            vecs = self.feature_vecs[self.df.validation.values]
            ds = tf.data.Dataset.from_tensor_slices((vecs, ys)
                                                      ).batch(batch_size)
        return self._strategy.experimental_distribute_dataset(ds)

        
    def build_training_step(self, weight_decay=0, opt_type="adam", lr=1e-3, 
                            lr_decay=0, decay_type="cosine", finetune=False):
        """
        
        """
        opt = build_optimizer(lr, lr_decay=lr_decay, opt_type=opt_type,
                              decay_type=decay_type)
        
            
        trainfunc = build_training_function(self.loss_fn, opt,
                            self.models["fine_tuning"],
                            self.models["output"],
                            feature_extractor=self.feature_extractor,
                            lam=self._model_params["semisup"]["fixmatch_weight_lambda"],
                            tau=self._model_params["semisup"]["fixmatch_tau"],
                            weight_decay=weight_decay,
                            finetune=finetune,
                            domain_weight=self._model_params["semisup"]["domain_confusion_weight_lambda"],
                            num_domains=self._num_domains)
        lam = self._model_params["semisup"]["fixmatch_weight_lambda"]
        domain_weight = self._model_params["semisup"]["domain_confusion_weight_lambda"]
        if (lam == 0)&(domain_weight == 0):
            @tf.function
            def training_step(x,y):
                per_example_losses = self._strategy.run(trainfunc, args=(x,y))

                losses = [self._strategy.reduce(
                    tf.distribute.ReduceOp.MEAN, 
                    k, axis=None) for k in per_example_losses]
                return losses
        elif (lam == 0)&(domain_weight > 0):
            @tf.function
            def training_step(x, y, x_unlab_wk, domain_labels):
                per_example_losses = self._strategy.run(trainfunc, 
                                                        args=(x,y, x_unlab_wk, None,
                                                              domain_labels))

                losses = [self._strategy.reduce(
                    tf.distribute.ReduceOp.MEAN, 
                    k, axis=None) for k in per_example_losses]
                return losses
        else:
            @tf.function
            def training_step(x, y, x_unlab_wk, x_unlab_str):
                per_example_losses = self._strategy.run(trainfunc, 
                                                        args=(x,y, x_unlab_wk, x_unlab_str))

                losses = [self._strategy.reduce(
                    tf.distribute.ReduceOp.MEAN, 
                    k, axis=None) for k in per_example_losses]
                return losses
        
        self._training_function = training_step

    def _run_one_training_epoch(self, batch_size=32, num_samples=None,
                                sampling="class", indexlist=None):
        """
        Run one training epoch
        """
        ds = self._training_dataset(batch_size, num_samples, sampling, indexlist)
        
        if self._semi_supervised or self._domain_adapt:
            #for (x, x_unlab), y in ds:
            # in semisupervised case- b is strongly-augmented data for FixMatch
            # in domain adaptation case- b is a label for the domain of each
            # image in unlab_wk
            for (x,y), (unlab_wk, b) in ds:
                loss, ss_loss = self._training_function(x, y, unlab_wk, b)
                self.training_loss.append(loss.numpy())
                self.semisup_loss.append(ss_loss.numpy())
        else:
            for x, y in ds:
                loss, ss_loss = self._training_function(x, y)
                self.training_loss.append(loss.numpy())
                self.semisup_loss.append(ss_loss.numpy())
                
    
    def _build_loss_tf_fn(self):
        # convenience function we'll use for computing test loss
        #@tf.function
        def meanloss(x,y):
            if self.models["feature_extractor"] is not None:
                x = self.models["feature_extractor"](x)
            x = self.models["fine_tuning"](x)
            y_pred = self.models["output"](x)
            loss = self.loss_fn(y, y_pred)
            return tf.reduce_mean(loss)
        
        @tf.function
        def meanloss_distributed(x,y):
                per_example_loss = self._strategy.run(meanloss, args=(x,y))
                return self._strategy.reduce(
                    tf.distribute.ReduceOp.MEAN, 
                    per_example_loss, axis=None)
        return meanloss_distributed
            
    
    def predict_on_all(self, batch_size=None):
        """
        Run inference on all the data; save to self.pred_df
        """
        if batch_size is None:
            batch_size = self._default_prediction_batch_size
        ds, num_steps = self._pred_dataset(batch_size)
        predictions = self.models["full"].predict(ds, steps=num_steps)

        self.pred_df.loc[:, self.classes] = predictions
        
    
    def predict_on_val(self, batch_size=None):
        """
        Run inference on validation data only; save to self.pred_df
        """
        if batch_size is None:
            batch_size = self._default_prediction_batch_size
        ds = self._val_dataset(batch_size)
        num_steps = int(np.ceil(len(self.df)/batch_size))
        predictions = self.models["full"].predict(ds, steps=num_steps)

        self.pred_df.loc[self.df.validation, self.classes] = predictions
    
    def _stratified_sample(self, N=None):
        return stratified_sample(self.df, N)
    
    def _load_img(self, f):
        """
        Wrapper for patchwork._util._load_img
        
        :f: string; path to file
        """
        return _load_img(f, norm=self._norm, num_channels=self._num_channels, 
                resize=self._imshape)
    
    def save(self):
        """
        Macro to write out labels, predictions, and models
        """
        if self._logdir is not None:
            self.df.to_csv(os.path.join(self._logdir, "labels.csv"), index=False)
            self.pred_df.to_csv(os.path.join(self._logdir, "predictions.csv"), index=False)
            for m in self.models:
                if self.models[m] is not None:
                    self.models[m].save(os.path.join(self._logdir, m+".h5"))
                    
    def compute_badge_embeddings(self, select_output=None):
        """
        Use a trained model to compute output-gradient vectors for the
        BADGE algorithm for active learning.
        
        Build a sampling object and record it at self._badge_sampler.
        
        Note that this stores all output gradients IN MEMORY.
        """
        # compute badge embeddings- define a tf.function for it
        compute_output_gradients = _build_output_gradient_function(self.models['full'],
                                                            select_output=select_output)
        # then run that function across all the iamges.
        output_gradients = np.concatenate(
            [compute_output_gradients(x).numpy() 
             for x in self._pred_dataset()[0]], axis=0)
        # find the indices that have already been fully 
        # labeled, so we can avoid sampling nearby
        indices = list(find_labeled_indices(self.df)) + \
                    list(find_excluded_indices(self.df))
        
        # initialize sampler
        self._badge_sampler = KPlusPlusSampler(output_gradients, indices=indices)
        
    
    def build_nearest_neighbor_adjacency_matrix(self, pred_batch_size=32, n_neighbors=10,
                                                temp=0.01):
        """
        
        """
        # build average-pool embedding model
        fcn = self.models["feature_extractor"]
        inpt = tf.keras.layers.Input((None, None, self._num_channels))
        net = fcn(inpt)
        net = tf.keras.layers.GlobalAvgPool2D()(net)
        model = tf.keras.Model(inpt, net)
        
        # generate predictions
        ds, steps = self._pred_dataset(pred_batch_size)
        features = model.predict(ds, steps=steps)
        
        # compute matrix
        self._adjacency_matrix = _get_weighted_adjacency_matrix(features, n_neighbors, temp)
    
   
    def build_projected_embeddings(self, pred_batch_size=32, proj_dim=128):
        """
        
        """
        # build average-pool embedding model
        fcn = self.models["feature_extractor"]
        inpt = tf.keras.layers.Input((None, None, self._num_channels))
        net = fcn(inpt)
        net = tf.keras.layers.GlobalAvgPool2D()(net)
        model = tf.keras.Model(inpt, net)
        
        D = model.output_shape[-1]
        
        # build a random projection matrix
        proj = np.random.normal(0, 1, size=(D, proj_dim))
        
        # generate predictions
        ds, steps = self._pred_dataset(pred_batch_size)
        
        matrix = np.concatenate([
            tf.nn.l2_normalize(model(x), 1).numpy().dot(proj)
            for x in ds
            ], 0)
        
        # find the indices that have already been fully 
        # labeled, so we can avoid sampling nearby
        indices = list(find_labeled_indices(self.df)) + \
                    list(find_excluded_indices(self.df))
        
        # initialize sampler
        self._diversity_sampler = KPlusPlusSampler(matrix, indices=indices)
        
    def _update_diversity_sampler(self):
        if self._diversity_sampler is not None:
            X = self._diversity_sampler.X
            indices = list(find_labeled_indices(self.df)) + \
                    list(find_excluded_indices(self.df))
            self._diversity_sampler = KPlusPlusSampler(X, indices=indices)  
        
     
   
    def propagate_labels(self, iterations=10):
        """
        
        """
        assert hasattr(self, "_adjacency_matrix"), "gotta compute adjacency matrix first!!"
        self.pred_df = _propagate_labels(self.df, self._adjacency_matrix,
                                         t=iterations)
    
    def random_hyperparameter_search(self, N, semisup_lambda=None, 
                                     learning_rate=None, finetune=None, 
                                     lr_decay=False,
                                     optimizer=False,
                                     samplingstrategy=False):
        """
        
        """
        for _ in tqdm(range(N)):
            
            if semisup_lambda is not None:
                s = semisup_lambda
                self.modelpicker._semisup["lambda"].value = float(np.random.uniform(s[0], 
                                                                  s[1], dtype=np.float32))
                
            if learning_rate is not None:
                l = learning_rate
                newval = np.exp(np.random.uniform(np.log(l[0]), np.log(l[1]),
                                                  dtype=np.float32))
                self.trainmanager._learn_rate.value = float(newval)
                
            if finetune is not None:
                f = finetune
                newval = np.random.randint(f[0], f[1])
                self.trainmanager._fine_tune_after.value = newval
                
            if lr_decay:
                newval = np.random.choice(["none", "cosine"])
                self.trainmanager._lr_decay.value = newval
                
            if optimizer:
                newval = np.random.choice(["adam", "momentum"])
                self.trainmanager._opt_chooser.value = newval
                
            if samplingstrategy:
                newval = np.random.choice(["class", "instance", "squareroot"])
                self.trainmanager._sample_chooser.value = newval
                
            
                
                
            self.modelpicker._build_callback()
            self.trainmanager._train_callback()
    
    
    