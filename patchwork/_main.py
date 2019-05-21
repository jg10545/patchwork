import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import warnings

prompt_txt = "Enter comma-delimited list of class-1 patches:"
EPSILON = 1e-5

class PatchWork(object):
    
    def __init__(self, feature_vecs, imfiles, epochs=1, min_count=10, epsilon=0, 
                 batch_size=64, verbose=0, df=None, **kwargs):
        """
        :feature_vecs: numpy array of feature data for each unlabeled training point
        :imfiles: list of strings of corresponding raw images
        :epochs: how many epochs to train for each iteration
        :min_count: minimum number of examples per class before network starts training
        :epsilon: epsilon-greedy hyperparameter. 0 for greedy sampling
        :batch_size: batch size for training
        :verbose: whether to print training information
        :df: dataframe of previously-labeled image metadata
        :kwargs: passed to model building function
        """
        self._batch_size = batch_size
        self._verbose = verbose
        self.counter = 0
        self._epsilon = epsilon
        self._min_count = min_count
        self._feature_vecs = feature_vecs
        self._imfiles = imfiles
        self.M = 16
        self.N = feature_vecs.shape[0]
        self._epochs = epochs
        # initialize model
        self.model = self._build_model(**kwargs)
        # initialize labels
        self.labels = np.array([np.nan for x in 
                                range(feature_vecs.shape[0])])
        self._sample_weights = np.ones(feature_vecs.shape[0])
        self._update_unlabeled()
        self.test_acc = []
        self.test_auc = []
        
        if df is not None:
            self._load_labels_from_df(df)
        
    def _update_unlabeled(self):
        # update our array keeping track of unlabeled images
        self.unlabeled_indices = np.arange(self.N)[np.isnan(self.labels)]
        
    def _build_model(self, inpt_shape=(6,6,1024), **kwargs):
        """
        Code to construct a tf.keras Model object
        
        :inpt_shape: tuple; shape of input tensor (neglecting batch size)
        """
        inpt = tf.keras.Input(shape=inpt_shape)
        net = tf.keras.layers.GlobalMaxPool2D("channels_last")(inpt)
        net = tf.keras.layers.Dense(1, activation=tf.keras.activations.sigmoid)(net)
        
        model = tf.keras.Model(inputs=inpt, outputs=net)
        model.compile(
            optimizer=tf.keras.optimizers.RMSprop(1e-3),
            #optimizer=tf.keras.optimizers.SGD(1e-3),
            loss=tf.keras.losses.binary_crossentropy,
            metrics=["accuracy"]
        )
        return model
        
    def _plot_sample(self, samp):
        """
        Plot a sample of images in a 4x4 grid
        """
        i = 1
        for s in samp:
            plt.subplot(4,4,i)
            plt.imshow(Image.open(self._imfiles[s]))
            plt.axis("off")
            plt.title(i, fontsize=14)
            i += 1
            
    def random_sample(self):
        return np.random.choice(self.unlabeled_indices, 
                                size=self.M, replace=False)
    
    def _get_input(self, prompt=prompt_txt):
        inpt = input(prompt)
        inpt = np.array([int(x.strip())-1 for x in inpt.split(",") 
                     if len(x) > 0])
        if (inpt < 0).any() or (inpt > 15).any():
            assert False, "what is this crap"
        return inpt
       
    def _training_generator(self, bs):
        N = self._feature_vecs.shape[0]
        pos = np.arange(N)[self.labels==1]
        neg = np.arange(N)[self.labels==0]
        all_inds = np.concatenate([pos,neg])
        probs = np.concatenate([(0.5/len(pos))*np.ones(len(pos)),
                        0.5/len(neg)*np.ones(len(neg))])
        while True:
            inds = np.random.choice(all_inds, size=bs, replace=True, p=probs)
            yield self._feature_vecs[inds,:], self.labels[inds], self._sample_weights[inds]

    
    def uncert_sample(self, epsilon=0):
        # compute probs for all vectors
        weights = np.ones(self.M)
        predictions = self.model.predict(self._feature_vecs)[:,-1]
        predictions[predictions == 0] = EPSILON
        predictions[predictions == 1] = 1-EPSILON
        # compute entropies
        H = -predictions*np.log2(predictions) - \
            (1-predictions)*np.log2(1-predictions)
        # highest-entropy unlabeled vectors
        highest_entropy = H[self.unlabeled_indices].argsort()[::-1]
        uncert_ind = self.unlabeled_indices[highest_entropy[:self.M]]
        # if epsilon > 0: do epsilon-greedy
        if epsilon > 0:
            num_random = np.random.binomial(16, epsilon)
            if num_random > 0: # otherwise code below breaks
                rand_ind = self.random_sample()
                randpicks = np.array([x for x in rand_ind if x not in uncert_ind[:-num_random]])
                uncert_ind = np.concatenate([uncert_ind[:-num_random], randpicks[:num_random]])
                weights[-num_random:] /= epsilon
                # shuffle order so it won't be as obvious which are random
                order = np.random.choice(np.arange(self.M), size=self.M, replace=False)
                uncert_ind = uncert_ind[order]
                weights = weights[order]
        return uncert_ind, weights 
    
    
    def iterate(self, groundtruth=None):
        assert len(self.unlabeled_indices) >= 16, "not enough unlabeled samples"
        # pick our indices- pull randomly if there aren't enough examples
        if ((self.labels == 0).sum() < self._min_count) or ((self.labels==1).sum() < self._min_count):
            sample = self.random_sample()
        # otherwise update model and do uncertainty sampling
        else:
            gen = self._training_generator(self._batch_size)
            self._hist = self.model.fit_generator(gen, steps_per_epoch=100, 
                                                  epochs=self._epochs, verbose=self._verbose)
            sample, weights = self.uncert_sample(self._epsilon)
            self._sample_weights[sample] = weights
            
        # plot the images
        if groundtruth is None:
            self._plot_sample(sample)
            plt.show()
        # get user feedback
        if groundtruth is None:
            positives = self._get_input()
        else:
            positives = np.array([s for s in np.arange(len(sample)) if (groundtruth[sample[s]]==1)])
        # update labels
        self.labels[sample] = 0
        if len(positives) > 1:
            self.labels[sample[positives]] = 1
        self._update_unlabeled()
        
        self.counter += 1

    def __call__(self, num_calls=1, groundtruth=None, testx=None, testy=None):
        for _ in tqdm(range(num_calls)):
            self.iterate(groundtruth)
            if (testx is not None) and (testy is not None):
                preds = self.model.predict(testx)
                self.test_auc.append(roc_auc_score(testy, preds[:,-1]))
                
    def export_labels(self):
        """
        Return labels as a pandas dataframe
        """
        labeled = ~np.isnan(self.labels)
        df = pd.DataFrame({"file":self._imfiles[labeled], 
                   "label":self.labels[labeled], 
                   "weights":self._sample_weights[labeled]})
        return df
    
    def _load_labels_from_df(self, df):
        """
        Update weights by importing previously-labeled data
        """
        imfiles = list(self._imfiles)
        for i, s in df.iterrows():
            if s["file"] in imfiles:
                ind = imfiles.index(s["file"])
                self.labels[ind] = s["label"]
                self._sample_weights[ind] = s["weights"]
            else:
                warnings.warn("%s not found")
            
        self._update_unlabeled()