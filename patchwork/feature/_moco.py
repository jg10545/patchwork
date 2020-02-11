# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf

from patchwork.feature._generic import GenericExtractor
from patchwork._augment import augment_function
from patchwork.loaders import _image_file_dataset



def copy_model(mod):
    """
    Clone a Keras model and set the new model's trainable weights to the old 
    model's weights
    """
    new_model = tf.keras.models.clone_model(mod)
    
    for orig, clone in zip(mod.trainable_variables, new_model.trainable_variables):
        clone.assign(orig)
    return new_model


def exponential_model_update(slow, fast, alpha=0.999):
    """
    Update the weights of a "slow" network as a single-exponential 
    average of a "fast" network. Return the sum of squared
    differences.
    
    :slow: Keras model containing exponentially-smoothed weights
    :fast: Keras model with same structure as slow, but updated more
        quickly from a different mechanism
    :alpha: exponential smoothing parameter
    """
    rolling_sum = 0
    for s, f in zip(slow.trainable_variables, fast.trainable_variables):
        rolling_sum += tf.reduce_sum(tf.square(s-f))
        s.assign(alpha*s + (1-alpha)*f)
    return rolling_sum



def build_augment_pair_dataset(imfiles, imshape=(256,256), batch_size=256, 
                      num_parallel_calls=None, norm=255,
                      num_channels=3, augment=True,
                      single_channel=False):
    """
    Build a tf.data.Dataset object for training momentum 
    contrast. Generates pairs of augmentations from a single
    image.
    """
    assert augment, "don't you need to augment your data?"
    _aug = augment_function(imshape, augment)
    
    ds = _image_file_dataset(imfiles, imshape=imshape, 
                             num_parallel_calls=num_parallel_calls,
                             norm=norm, num_channels=num_channels,
                             shuffle=True, single_channel=single_channel)  
    
    a1 = ds.map(_aug, num_parallel_calls=num_parallel_calls)
    a2 = ds.map(_aug, num_parallel_calls=num_parallel_calls)
   
    ds = ds.zip((a1, a2))
    ds = ds.batch(batch_size)
    ds = ds.prefetch(1)
    return ds


def build_momentum_contrast_training_step(fcn, mo_fcn, optimizer, buffer, batches_in_buffer, alpha=0.999, tau=0.07):
    """
    Function to build tf.function for a MoCo training step. Basically just follow
    Algorithm 1 in He et al's paper.
    """
    flatten = tf.keras.layers.GlobalAvgPool2D()
    
    @tf.function
    def training_step(img1, img2, step):
        print("tracing training step")
        batch_size = img1.shape[0]
        with tf.GradientTape() as tape:
            # compute averaged and normalized embeddings for each 
            # separately-augmented batch of pairs of images. Each is (N,d)
            q = tf.nn.l2_normalize(flatten(fcn(img1)), axis=1)
            k = tf.nn.l2_normalize(flatten(mo_fcn(img2)), axis=1)
    
            # compute positive logits- (N,1)
            positive_logits = tf.squeeze(
                tf.matmul(tf.expand_dims(q,1), 
                      tf.expand_dims(k,1), transpose_b=True),
                axis=-1)
            # and negative logits- (N, buffer_size)
            negative_logits = tf.matmul(q, buffer, transpose_b=True)
            # assemble positive and negative- (N, buffer_size+1)
            all_logits = tf.concat([positive_logits, negative_logits], axis=1)
            # create labels (correct class is 0)- (N,)
            labels = tf.zeros((batch_size,), dtype=tf.int32)
            # compute crossentropy loss
            loss = tf.reduce_mean(
                    tf.nn.sparse_softmax_cross_entropy_with_logits(
                            labels, all_logits/tau))
    
        # update fast model
        variables = fcn.trainable_variables
        gradients = tape.gradient(loss, variables)
        optimizer.apply_gradients(zip(gradients, variables))
        # update slow model
        weight_diff = exponential_model_update(mo_fcn, fcn, alpha)
    
        # update buffer
        i = step % batches_in_buffer
        _ = buffer[batch_size*i:batch_size*(i+1),:].assign(k)
        
        return loss, weight_diff
    return training_step




class MomentumContrastTrainer(GenericExtractor):
    """
    Class for training a Momentum Contrast model.
    
    Based on "Momentum Contrast for Unsupervised Visual Representation 
    Learning" by He et al.
    """

    def __init__(self, logdir, trainingdata, testdata=None, fcn=None, 
                 augment=True, batches_in_buffer=10, alpha=0.999, 
                 tau=0.07,
                 lr=0.01, lr_decay=100000,
                 imshape=(256,256), num_channels=3,
                 norm=255, batch_size=64, num_parallel_calls=None,
                 sobel=False, single_channel=False, notes="",
                 downstream_labels=None):
        """
        :logdir: (string) path to log directory
        :trainingdata: (list) list of paths to training images
        :testdata: (list) filepaths of a batch of images to use for eval
        :fcn: (keras Model) fully-convolutional network to train as feature extractor
        :augment: (dict) dictionary of augmentation parameters, True for defaults
        :batches_in_buffer:
        :alpha:
        :tau:
        :lr: (float) initial learning rate
        :lr_decay: (int) steps for learning rate to decay by half (0 to disable)
        :imshape: (tuple) image dimensions in H,W
        :num_channels: (int) number of image channels
        :norm: (int or float) normalization constant for images (for rescaling to
               unit interval)
        :batch_size: (int) batch size for training
        :num_parallel_calls: (int) number of threads for loader mapping
        :sobel: whether to replace the input image with its sobel edges
        :single_channel: if True, expect a single-channel input image and 
                stack it num_channels times.
        :notes: (string) any notes on the experiment that you want saved in the
                config.yml file
        :downstream_labels: dictionary mapping image file paths to labels
        """
        assert augment is not False, "this method needs an augmentation scheme"
        self.logdir = logdir
        self.trainingdata = trainingdata
        self._downstream_labels = downstream_labels
        channels = 3 if sobel else num_channels
        
        self._file_writer = tf.summary.create_file_writer(logdir, flush_millis=10000)
        self._file_writer.set_as_default()
        
        # if no FCN is passed- build one
        if fcn is None:
            fcn = tf.keras.applications.ResNet50V2(weights=None, include_top=False)
        self.fcn = fcn
        momentum_encoder = copy_model(fcn)
        self._models = {"fcn":fcn, "momentum_encoder":momentum_encoder}
        
        # build training dataset
        self._ds = build_augment_pair_dataset(trainingdata, 
                            imshape=imshape, batch_size=batch_size,
                            num_parallel_calls=num_parallel_calls, 
                            norm=norm, num_channels=channels, 
                            augment=augment, single_channel=single_channel)
        
        # create optimizer
        if lr_decay > 0:
            learnrate = tf.keras.optimizers.schedules.ExponentialDecay(lr, 
                                            decay_steps=lr_decay, decay_rate=0.5,
                                            staircase=False)
        else:
            learnrate = lr
        self._optimizer = tf.keras.optimizers.SGD(learnrate, momentum=0.9)
        
        # build buffer
        K = batch_size*batches_in_buffer
        d = fcn.output_shape[-1]
        self._buffer = tf.Variable(np.zeros((K,d), dtype=np.float32))
        
        # build training step
        self._training_step = build_momentum_contrast_training_step(
                fcn, 
                momentum_encoder, 
                self._optimizer, 
                self._buffer, 
                batches_in_buffer, alpha, tau)
        # build evaluation dataset
        #if testdata is not None:
        #    self._test_ds, self._test_steps = dataset(testdata,
        #                             imshape=imshape,norm=norm,
        #                             sobel=sobel, num_channels=num_channels,
        #                             single_channel=single_channel)
        #    self._test = True
        #else:
        #    self._test = False
        self._test = False
        self._test_labels = None
        self._old_test_labels = None
        
        # build prediction dataset for clustering

        self.step = 0
        self._step_var = tf.Variable(0, dtype=tf.int64)
        
        # parse and write out config YAML
        self._parse_configs(augment=augment, 
                            batches_in_buffer=batches_in_buffer, 
                            alpha=alpha, tau=tau, lr=lr, lr_decay=lr_decay, 
                            imshape=imshape, num_channels=num_channels,
                            norm=norm, batch_size=batch_size,
                            num_parallel_calls=num_parallel_calls, sobel=sobel,
                            single_channel=single_channel, notes=notes)
        
        
    def _run_training_epoch(self, **kwargs):
        """
        
        """
        for x, y in self._ds:
            loss, weight_diff = self._training_step(x,y, self._step_var)
            
            self._record_scalars(loss=loss, weight_diff=weight_diff)
            
            self._step_var.assign_add(1)
            self.step += 1
            
 
    def evaluate(self):
        b = tf.expand_dims(tf.expand_dims(self._buffer,0),-1)
        self._record_images(buffer=b)
            
        if self._downstream_labels is not None:
            # choose the hyperparameters to record
            if not hasattr(self, "_hparams_config"):
                from tensorboard.plugins.hparams import api as hp
                hparams = {
                    hp.HParam("tau", hp.RealInterval(0., 10000.)):self.config["tau"],
                    hp.HParam("alpha", hp.RealInterval(0., 1.)):self.config["alpha"],
                    hp.HParam("batches_in_buffer", hp.IntInterval(1, 1000000)):self.config["batches_in_buffer"],
                    hp.HParam("sobel", hp.Discrete([True, False])):self.input_config["sobel"]
                    }
            else:
                hparams=None
            self._linear_classification_test(hparams)
        
            
        
