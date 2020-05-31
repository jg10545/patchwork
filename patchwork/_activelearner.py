# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import tensorflow as tf

from patchwork._sample import stratified_sample, find_labeled_indices, find_excluded_indices
from patchwork._badge import KPlusPlusSampler, _build_output_gradient_function
from patchwork._losses import masked_binary_crossentropy
from patchwork._labeler import pick_indices

class ActiveLearner():
    """
    Wrapper class for basic human-in-the-loop active learning. Assumes precomputed
    features are stored in memory.
    """
    def __init__(self, features, hidden=[128], samples_per_epoch=10000,
                 batch_size=16, lr=1e-3, use_badge=True, label_dicts=None):
        """
        :features: 2D numpy array, (num_tiles, feature_dim)
        :hidden: list of ints; number of neurons in each hidden layer between
            features and output
        :samples_per_epoch: int; how many samples to take each time you call train()
        :batch_size: int; batch size for training
        :lr: float; learning rate for training
        :use_badge: bool; whether to precompute BADGE gradient embeddings. This will add some
            time to train() calls but allows you to use diversity sampling.
        :label_dicts: list of dicts; optionally instantiate ActiveLearner with existing tags.
        """
        self.features = features
        self.N = features.shape[0]
        self.d = features.shape[1]
        self._hidden = hidden
        self._train_params = {"samples_per_epoch":samples_per_epoch, "batch_size":batch_size,
                             "lr":lr}
        self._use_badge = use_badge
        if label_dicts is None:
            label_dicts = [{"filepath":None, "exclude":False, 
                              "validation":False} for _ in range(self.N)]
        self._label_dicts = label_dicts
        
        self.training_loss = []
        self.pred_df = pd.DataFrame(np.zeros(self.N), columns=["foo"])
        self.loss = []
        self.classes = []
        
    def add_classes(self, *categories):
        """
        Add one or more categories for classification.
        """
        for c in categories:
            if c not in self.classes:
                self.classes.append(c)
        
    def _build_model(self, num_classes, lr=1e-3):
        inpt = tf.keras.layers.Input((self.d))
        net = inpt
        for h in self._hidden:
            net = tf.keras.layers.Dense(h, activation="relu")(net)
        fine_tuning_model = tf.keras.Model(inpt, net)
        
        if len(self._hidden) > 0:
            inpt_o = tf.keras.layers.Input((self._hidden[-1]))
        else:
            inpt_o = tf.keras.layers.Input((self.d))
        
        net = inpt_o
        net = tf.keras.layers.Dense(num_classes, activation="sigmoid")(net)
        output_model = tf.keras.Model(inpt_o, net)
        
        model = tf.keras.Model(inpt, output_model(fine_tuning_model(inpt)))
        self._fine_tuning_model = fine_tuning_model
        self._output_model = output_model
        return model
    
    def _build_training_dataset(self, df, num_classes, samples_per_epoch):
        indices, labels = stratified_sample(df, samples_per_epoch, 
                                            return_indices=True)
        ds = tf.data.Dataset.from_tensor_slices((self.features[indices], labels))
        ds = ds.batch(self._train_params["batch_size"])
        ds = ds.prefetch(1)
        return ds

    def _compute_badge_embeddings(self):
        """
        Use a trained model to compute output-gradient vectors for the
        BADGE algorithm for active learning.
        
        Build a sampling object and record it at self._badge_sampler.
        
        Note that this stores all output gradients IN MEMORY.
        """
        # compute badge embeddings- define a tf.function for it
        compute_output_gradients = _build_output_gradient_function(
                                        self._fine_tuning_model,
                                        self._output_model)
        # then run that function across all the iamges.
        output_gradients = tf.map_fn(compute_output_gradients, self.features).numpy()
        # find the indices that have already been fully or partially
        # labeled, so we can avoid sampling nearby. Also don't let it sample
        # anything we explicitly excluded.
        df = pd.DataFrame(self._label_dicts, columns=["filepath", "exclude", "validation"]+self.classes)
        indices = list(find_labeled_indices(df)) + list(find_excluded_indices(df))
        
        # initialize sampler
        self._badge_sampler = KPlusPlusSampler(output_gradients, indices=indices)
    
    def train(self, samples_per_epoch=None):
        """
        Hey now, you're an all-star. Get your train on.
        
        :samples_per_epoch: override the number of samples.
        """
        if samples_per_epoch is None: samples_per_epoch = self._train_params["samples_per_epoch"]
        # convert labels to a dataframe
        df = pd.DataFrame(self._label_dicts)
        num_classes = len(self.classes)
        # build a dataset for training
        ds = self._build_training_dataset(df, num_classes, samples_per_epoch)
        # initialize a model and optimizer
        self.model = self._build_model(num_classes)
        opt = tf.keras.optimizers.Adam(self._train_params["lr"])
        
        # train the model, recording loss at each step
        @tf.function
        def training_step(x,y):
            with tf.GradientTape() as tape:
                pred = self.model(x)
                loss = masked_binary_crossentropy(y, pred)
            variables = self.model.trainable_variables
            grads = tape.gradient(loss, variables)
            opt.apply_gradients(zip(grads, variables))
            return loss
        
        for x, y in ds:
            self.loss.append(training_step(x,y).numpy())
            
        # update predictions
        preds = self.model.predict(self.features)
        self.pred_df = pd.DataFrame(preds, columns=self.classes)  
        # if we're planning on doing BADGE sampling, precompute embeddings
        if self._use_badge:
            self._compute_badge_embeddings()
        
                
            
            
    def tag(self, index, **tags):
        """
        Update the label dictionary for one tile.
        
        :index: index of the tile to update
        :tags: key-value pairs of categories to update.
        """
        for t in tags:
            if t in ["exclude", "validation"]:
                assert tags[t] in [True, False], "exclude and validtion tags should be Boolean"
            else:
                assert tags[t] in [0,1, None], "tags should be 0, 1, or None"
            assert t in self.classes+["exclude", "validation"], "unknown category %s"%t
            self._label_dicts[index][t] = tags[t]
            
    def sample_random(self, num_samples=5, subset_by="unlabeled"):
        """
        Return randomly-sampled indices
        
        :num_samples: how many to return
        :subset_by: string; which subset of the data to sample from: "unlabeled",
            "fully labeled", or "partially labeled"
        """
        df = pd.DataFrame(self._label_dicts)
        return pick_indices(df, self.pred_df, num_samples,
                                       sort_by="random", subset_by=subset_by)
    
    
    def sample_uncertainty(self, num_samples=5, subset_by="unlabeled"):
        """
        Return uncertainty-sampled indices; the feature vectors that the
        model is least certain about. Must have a trained model first!
        
        :num_samples: how many to return
        :subset_by: string; which subset of the data to sample from: "unlabeled",
            "fully labeled", or "partially labeled"
        """
        assert len(self.loss) > 0, "train a model first"
        df = pd.DataFrame(self._label_dicts)
        return pick_indices(df, self.pred_df, num_samples,
                                       sort_by="max entropy", subset_by=subset_by)
    
    
    def sample_diversity(self, num_samples=5):
        """
        Return indices sampled using the BADGE algorithm to try and give as much variety
        in the results as possible, biased by model uncertainty. Must have a trained model
        first and have use_badge=True.
        
        Only samples from unlabeled tiles.
        
        :num_samples: how many to return
        """
        assert hasattr(self,"_badge_sampler"), "train a model with use_badge=True first"
        return self._badge_sampler.choose(num_samples)
    
    def highest_predicted(self, category, num_samples=5, subset_by="unlabeled"):
        """
        Return indices of tiles with the highest predicted probability for a particular
        category.
        
        :category category to check
        :num_samples: how many indices to return
        :subset_by: string; which subset of the data to sample from: "unlabeled",
            "fully labeled", or "partially labeled"
        """
        assert len(self.loss) > 0, "train a model first"
        assert category in self.classes, "category %s not found"%category
        df = pd.DataFrame(self._label_dicts)
        return pick_indices(df, self.pred_df, num_samples,
                                       sort_by="high: %s"%category, subset_by=subset_by)
    
    def lowest_predicted(self, category, num_samples=5, subset_by="unlabeled"):
        """
        Return indices of tiles with the lowest predicted probability for a particular
        category.
        
        :category category to check
        :num_samples: how many indices to return
        :subset_by: string; which subset of the data to sample from: "unlabeled",
            "fully labeled", or "partially labeled"
        """
        assert len(self.loss) > 0, "train a model first"
        assert category in self.classes, "category %s not found"%category
        df = pd.DataFrame(self._label_dicts)
        return pick_indices(df, self.pred_df, num_samples,
                                       sort_by="low: %s"%category, subset_by=subset_by)
    
    def export_tags_to_json(self):
        """
        Return label dictionary in a JSON compatible format
        """
        return [{k:d[k] for k in d if d[k] is not None} for
               d in self._label_dicts]