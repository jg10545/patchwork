import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K

from patchwork.loaders import _image_file_dataset#, dataset
from patchwork.feature._generic import GenericExtractor
from patchwork._augment import augment_function


eps = K.epsilon()

@tf.function
def compute_mutual_information(x,y):
    # compute P(x,y) as the outer product of two vectors, then sum over the batch
    P = tf.reduce_sum(
        tf.matmul(tf.expand_dims(x,2), tf.expand_dims(y,2), transpose_b=True),
    axis=0)
    # make P symmetric
    P = ((P + tf.transpose(P))/2)
    # make P is positive definite and renormalize
    P = tf.maximum(P, eps)
    P = P/tf.reduce_sum(P)
    # compute marginals  P(x) and P(y)
    Pi = tf.expand_dims(tf.reduce_sum(P,0),0)
    Pj = tf.expand_dims(tf.reduce_sum(P,1),1)
    # return mutual information in nats
    conditional_entropy = -1*tf.reduce_sum(P * (tf.math.log(P) - tf.math.log(Pi)))
    entropy = -1*tf.reduce_sum(P*tf.math.log(Pj))
    mutual_info = entropy - conditional_entropy
    return mutual_info, entropy, conditional_entropy

def compute_p(f_x, f_y, head):
    """
    Compute the full P matrix (for tensorboard display)
    """
    x = head(f_x)
    y = head(f_y)
    P = tf.reduce_sum(
            tf.matmul(tf.expand_dims(x,2), 
                     tf.expand_dims(y,2), transpose_b=True),
        axis=0)
    # make P symmetric
    P = ((P + tf.transpose(P))/2)
    # summarywriter needs a rank-4 tensor
    P = tf.expand_dims(tf.expand_dims(P,0),3).numpy()
    # for display- rescale so max value is 1
    P = P/np.max(P)
    return P  

def iic_training_step(x1, x2, fcn, heads, opt, variables, entropy_weight=0):
    """
    Training step for invariant information clustering.
    
    Initially built this with @tf.function decorator; kept running
    into "tf.function-decorated function tried to create variables on 
    non-first call" errors.
    
    :x1: batch of images
    :x2: batch of augmented images paired with x1
    :fcn: feature extractor
    :heads: list of output heads
    :opt: keras optimizer
    
    Returns mutual information loss
    """    
    with tf.GradientTape() as tape:
        features1 = fcn(x1)
        features2 = fcn(x2)
        
        loss = 0
        entropy = 0
        conditional_entropy = 0
        for h in heads:
            z1 = h(features1)
            z2 = h(features2)
            mutual_info, H, H_c = compute_mutual_information(z1,z2)
            loss -= mutual_info
            if entropy_weight > 0:
                loss -= entropy_weight*H
            
            entropy += H
            conditional_entropy += H_c
            
    gradients = tape.gradient(loss, variables)
    opt.apply_gradients(zip(gradients, variables))
    return loss, entropy, conditional_entropy


def build_iic_dataset(imfiles, r=5, imshape=(256,256), batch_size=256, 
                      num_parallel_calls=None, norm=255,
                      num_channels=3, augment=True,
                      single_channel=False):
    """
    Build a tf.data.Dataset object for training IIC.
    """
    assert augment, "don't you need to augment your data?"
    _aug = augment_function(imshape, augment)
    
    ds = _image_file_dataset(imfiles, imshape=imshape, 
                             num_parallel_calls=num_parallel_calls,
                             norm=norm, num_channels=num_channels,
                             shuffle=True, single_channel=single_channel)
    
    if r > 1:
        ds = ds.flat_map(lambda x: tf.data.Dataset.from_tensors(x).repeat(r))
        
    augmented_ds = ds.map(_aug, num_parallel_calls=num_parallel_calls)
    
    ds = ds.zip((ds, augmented_ds))
    ds = ds.batch(batch_size)
    ds = ds.prefetch(1)
    return ds
    
    



class InvariantInformationClusteringTrainer(GenericExtractor):
    """
    Class for training an IIC model
    """

    def __init__(self, logdir, trainingdata, testdata=None, fcn=None, 
                 augment=True, k=10, h=5, k_oc=25, r=5,
                 entropy_weight=0, lr=1e-4, lr_decay=100000,
                 imshape=(256,256), num_channels=3,
                 norm=255, batch_size=64, num_parallel_calls=None,
                 sobel=False, single_channel=False, notes="",
                 downstream_labels=None):
        """
        :logdir: (string) path to log directory
        :trainingdata: (list) list of paths to training images
        :testdata: (list) filepaths of a batch of images to use for eval
        :fcn: (keras Model) fully-convolutional network to train as feature extractor
        :augment: (dict) dictionary of augmentation parameters, True for defaults or
                False to disable augmentation
        :k: output dimension for clustering head
        :h: number of clustering sub-heads
        :k_oc: output dimension for overclustering head
        :entropy_weight: additional weight factor for entropy in loss function (see
                section 5 of supplementary material)
        :r: number of times to repeat each image sequentially in the data pipeline
        :lr: (float) initial learning rate
        :lr_decay: (int) steps for learning rate to decay by half (0 to disable)
        :imshape: (tuple) image dimensions in H,W
        :num_channels: (int) number of image channels
        :norm: (int or float) normalization constant for images (for rescaling to
               unit interval)
        :batch_size: (int) batch size for training
        :num_parallel_calls: (int) number of threads for loader mapping
        :sobel:
        :single_channel: if True, expect a single-channel input image and 
                stack it num_channels times.
        :notes: (string) any notes on the experiment that you want saved in the
                config.yml file
        :downstream_labels: dictionary mapping image file paths to labels
        """
        if sobel: assert False, "NOT YET IMPLEMENTED"
        self.logdir = logdir
        self.trainingdata = trainingdata
        self._downstream_labels = downstream_labels
        
        self._train_ds = build_iic_dataset(trainingdata, r=r, imshape=imshape,
                                           batch_size=batch_size, 
                                           num_parallel_calls=num_parallel_calls,
                                           norm=norm, num_channels=num_channels,
                                           augment=augment,
                                           single_channel=single_channel)
        if testdata is not None:
            self._test_ds = build_iic_dataset(testdata, r=r, imshape=imshape,
                                           batch_size=batch_size, 
                                           num_parallel_calls=num_parallel_calls,
                                           norm=norm, num_channels=num_channels,
                                           augment=augment,
                                           single_channel=single_channel)
            self._test = True
        else:
            self._test = False
        
        self._file_writer = tf.summary.create_file_writer(logdir, flush_millis=10000)
        self._file_writer.set_as_default()
        
        # if no FCN is passed- build one
        if fcn is None:
            assert False, "no default FCN implemented for IIC yet- try again, sucker"
        self.fcn = fcn
        self._models = {"fcn":fcn}   
        self._flat_model = tf.keras.Sequential([
            fcn,
            tf.keras.layers.GlobalAveragePooling2D()
        ])
        # initialize the clustering heads
        self.heads = [tf.keras.layers.Dense(k, activation="softmax")
                      for _ in range(h)]
        self.heads_oc = [tf.keras.layers.Dense(k_oc, activation="softmax")]
        self.flatten = tf.keras.layers.GlobalAveragePooling2D()
        # record trainable variables
        self._variables = self.fcn.trainable_variables
        for head in self.heads:
            self._variables += head.trainable_variables
        self._variables_oc = self.fcn.trainable_variables
        for head in self.heads_oc:
            self._variables_oc += head.trainable_variables
        
        # create optimizers
        if lr_decay > 0:
            learnrate = tf.keras.optimizers.schedules.ExponentialDecay(lr, 
                                            decay_steps=lr_decay, decay_rate=0.5,
                                            staircase=False)
        else:
            learnrate = lr
        self._optimizer = tf.keras.optimizers.Adam(learnrate)
        self._optimizer_oc = tf.keras.optimizers.Adam(learnrate)
        
        self.step = 0
        self._parse_configs(k=k, k_oc=k_oc, h=h, lr=lr, lr_decay=lr_decay,
                            r=r, entropy_weight=entropy_weight,
                            imshape=imshape, num_channels=num_channels,
                            norm=norm, batch_size=batch_size, 
                            num_parallel_calls=num_parallel_calls,
                            augment=augment, sobel=sobel, single_channel=single_channel,
                            notes=notes)
        
        
    @tf.function
    def model_train_step(self, x1, x2):
        return iic_training_step(x1, x2, self._flat_model,
                                 self.heads,
                                 self._optimizer, self._variables,
                                 self.config["entropy_weight"])
    
    @tf.function
    def ocmodel_train_step(self, x1,x2):
        return iic_training_step(x1, x2, self._flat_model, 
                                 self.heads_oc,
                                 self._optimizer_oc, self._variables_oc,
                                 self.config["entropy_weight"])

    
    def _run_training_epoch(self, **kwargs):
        """
        
        """
        # alternating epoch training
        for x1, x2 in self._train_ds:
            loss, H, H_c = self.model_train_step(x1,x2)
            self._record_scalars(model_loss=loss, entropy=H, 
                                 conditional_entropy=H_c)
            self.step += 1
            
        for x1, x2 in self._train_ds:
            loss, H, H_c = self.ocmodel_train_step(x1,x2)
            self._record_scalars(ocmodel_loss=loss, oc_entropy=H, 
                                 oc_conditional_entropy=H_c)
            self.step += 1
                
 
    def evaluate(self):
        if self._test:
            # compute features- in memory for now
            f_x = []
            f_y = []
            for x, y in self._test_ds:
                f_x.append(self.flatten(self.fcn(x)))
                f_y.append(self.flatten(self.fcn(y)))
            f_x = tf.concat(f_x, 0)
            f_y = tf.concat(f_y, 0)
            # compute P for each head
            for e, h in enumerate(self.heads):
                P = compute_p(f_x, f_y, h)
                self._record_images(**{"P_head_%s"%e:P})
                
            for e, h in enumerate(self.heads_oc):
                P = compute_p(f_x, f_y, h)
                self._record_images(**{"P_oc_head_%s"%e:P})
                
        if self._downstream_labels is not None:
            # choose the hyperparameters to record
            if not hasattr(self, "_hparams_config"):
                from tensorboard.plugins.hparams import api as hp
                hparams = {
                    hp.HParam("k", hp.IntInterval(0, 1000000), 
                              description="output dimension for clustering head"):self.config["k"],
                    hp.HParam("h", hp.IntInterval(0, 1000000), 
                              description="number of clustering sub-heads"):self.config["h"],
                    hp.HParam("k_oc", hp.IntInterval(0, 1000000), 
                              description="output dimension for overclustering head"):self.config["k_oc"],
                    hp.HParam("entropy_weight", hp.RealInterval(0., 1000000.), 
                              description="additional weight factor for entropy in loss function"):self.config["entropy_weight"],
                    hp.HParam("r", hp.IntInterval(0, 1000000), 
                              description="number of times to repeat each image sequentially in the data pipeline"):self.config["r"],
                    hp.HParam("sobel", hp.Discrete([True, False]),
                              description="whether Sobel filtering was applied to inputs"):self.input_config["sobel"]
                    }
            else:
                hparams=None
            self._linear_classification_test(hparams)
                
            