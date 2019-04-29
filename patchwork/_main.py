import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
from tqdm import tqdm

prompt_txt = "Enter comma-delimited list of class-1 patches:"
EPSILON = 1e-5

class PatchWork(object):
    
    def __init__(self, feature_vecs, imfiles, epochs=100, min_count=10):
        """
        :feature_vecs: numpy array of feature data for each unlabeled training point
        :imfiles: list of strings of corresponding raw images
        :epochs: how many epochs to train for each iteration
        :min_count: minimum number of examples per class before network starts training
        """
        self.counter = 0
        self._min_count = min_count
        self._feature_vecs = feature_vecs
        self._imfiles = imfiles
        self.M = 16
        self.N = feature_vecs.shape[0]
        self._epochs = epochs
        # initialize model
        self.model = self._build_model()
        # initialize labels
        self.labels = np.array([np.nan for x in 
                                range(feature_vecs.shape[0])])
        self._sample_weights = np.ones(feature_vecs.shape[0])
        self._update_unlabeled()
        self.test_acc = []
        
    def _update_unlabeled(self):
        # update our array keeping track of unlabeled images
        self.unlabeled_indices = np.arange(self.N)[np.isnan(self.labels)]
        
    def _build_model(self, inpt_shape=(6,6,1024)):
        """
        Code to construct a tf.keras Model object
        
        :inpt_shape: tuple; shape of input tensor (neglecting batch size)
        """
        inpt = tf.keras.Input(shape=inpt_shape)
        net = tf.keras.layers.GlobalMaxPool2D("channels_last")(inpt)
        net = tf.keras.layers.Dense(1, activation=tf.keras.activations.sigmoid)(net)
        
        model = tf.keras.Model(inputs=inpt, outputs=net)
        model.compile(
            optimizer=tf.keras.optimizers.SGD(1e-3),
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
            
    def _training_set(self):
        labeled = ~np.isnan(self.labels)
        return self._feature_vecs[labeled, :], self.labels[labeled]
    
    def uncert_sample(self):
        # compute probs for all vectors
        predictions = self.model.predict(self._feature_vecs).ravel()#[:,1]
        predictions[predictions == 0] = EPSILON
        predictions[predictions == 1] = 1-EPSILON
        # compute entropies
        H = -predictions*np.log2(predictions) - \
            (1-predictions)*np.log2(1-predictions)
        # highest-entropy unlabeled vectors
        highest_entropy = H[self.unlabeled_indices].argsort()[::-1]
        return self.unlabeled_indices[highest_entropy[:self.M]], predictions
    
    def iterate(self, groundtruth=None):
        assert len(self.unlabeled_indices) >= 16, "not enough unlabeled samples"
        # pick our indices- pull randomly if there aren't enough examples
        if ((self.labels == 0).sum() < self._min_count) or ((self.labels==1).sum() < self._min_count):
            sample = self.random_sample()
        # otherwise update model and do uncertainty sampling
        else:
            x,y = self._training_set()
            batch_size = min(x.shape[0], 64)
            self._hist = self.model.fit(x, y, batch_size=batch_size, 
                           epochs=self._epochs, verbose=0)#, sample_weight=self._sample_weights)
            sample, preds = self.uncert_sample()

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
        self.labels[sample[positives]] = 1
        self._update_unlabeled()
        
        self.counter += 1
        
    def __call__(self, num_calls=1, groundtruth=None, testx=None, testy=None):
        for _ in tqdm(range(num_calls)):
            self.iterate(groundtruth)
            if (testx is not None) and (testy is not None):
                acc = self.model.evaluate(testx, testy, verbose=0)
                self.test_acc.append(acc[-1])
            