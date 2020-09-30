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



def _build_augment_pair_dataset(imfiles, imshape=(256,256), batch_size=256, 
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
    
    def _augment_pair(x):
        return _aug(x), _aug(x)

    ds = ds.map(_augment_pair, num_parallel_calls=num_parallel_calls)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(1)
    return ds


def build_momentum_contrast_training_step(model, mo_model, optimizer, buffer, batches_in_buffer, alpha=0.999, tau=0.07):
    """
    Function to build tf.function for a MoCo training step. Basically just follow
    Algorithm 1 in He et al's paper.
    """
    
    @tf.function
    def training_step(img1, img2, step):
        print("tracing training step")
        batch_size = img1.shape[0]
        # compute averaged embeddings. tensor is (N,d)
        #k = tf.nn.l2_normalize(mo_model(img2, training=True), axis=1)
        # my shot at an alternative to shuffling BN
        a = int(batch_size/4)
        b = int(batch_size/2)
        c = int(3*batch_size/4)
        
        k1 = mo_model(tf.concat([img2[:a,:,:,:], img2[c:,:,:,:]], 0),
                      training=True)
        k2 = mo_model(img2[a:c,:,:,:], training=True)
        k = tf.nn.l2_normalize(tf.concat([
                k1[:a,:], k2, k1[a:,:]], axis=0
            ), axis=1)
        #print("k:", k.shape)
        with tf.GradientTape() as tape:
            # compute normalized embeddings for each 
            # separately-augmented batch of pairs of images. tensor is (N,d)
            #q = tf.nn.l2_normalize(model(img1, training=True), axis=1)
            q1 = model(img1[:b,:,:,:], training=True)
            q2 = model(img1[b:,:,:,:], training=True)
            q = tf.nn.l2_normalize(tf.concat([q1,q2], 0), axis=1)
            #print("q", q.shape)
            # compute positive logits- (N,1)
            positive_logits = tf.squeeze(
                tf.matmul(tf.expand_dims(q,1), 
                      tf.expand_dims(k,1), transpose_b=True),
                axis=-1)
            #print("positive logits", positive_logits.shape)
            # and negative logits- (N, buffer_size)
            negative_logits = tf.matmul(q, buffer, transpose_b=True)
            #print("negative_logits", negative_logits.shape)
            # assemble positive and negative- (N, buffer_size+1)
            all_logits = tf.concat([positive_logits, negative_logits], axis=1)
            #print("all_logits", all_logits.shape)
            # create labels (correct class is 0)- (N,)
            labels = tf.zeros((batch_size,), dtype=tf.int32)
            #print("labels", labels.shape)
            # compute crossentropy loss
            loss = tf.reduce_mean(
                    tf.nn.sparse_softmax_cross_entropy_with_logits(
                            labels, all_logits/tau))
    
        # update fast model
        variables = model.trainable_variables
        gradients = tape.gradient(loss, variables)
        optimizer.apply_gradients(zip(gradients, variables))
        # update slow model
        weight_diff = exponential_model_update(mo_model, model, alpha)
    
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
                 tau=0.07, output_dim=128, num_hidden=2048,
                 lr=0.01, lr_decay=100000, decay_type="exponential",
                 imshape=(256,256), num_channels=3,
                 norm=255, batch_size=64, num_parallel_calls=None,
                 single_channel=False, notes="",
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
        :output_dim:
        :num_hidden:
        :lr: (float) initial learning rate
        :lr_decay: (int) steps for learning rate to decay by half (0 to disable)
        :decay_type:
        :imshape: (tuple) image dimensions in H,W
        :num_channels: (int) number of image channels
        :norm: (int or float) normalization constant for images (for rescaling to
               unit interval)
        :batch_size: (int) batch size for training
        :num_parallel_calls: (int) number of threads for loader mapping
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
        
        self._file_writer = tf.summary.create_file_writer(logdir, flush_millis=10000)
        self._file_writer.set_as_default()
        
        # if no FCN is passed- build one
        if fcn is None:
            fcn = tf.keras.applications.ResNet50V2(weights=None, include_top=False)
        self.fcn = fcn
        # from "technical details" in paper- after FCN they did global pooling
        # and then a dense layer. i assume linear outputs on it.
        inpt = tf.keras.layers.Input((None, None, num_channels))
        features = fcn(inpt)
        pooled = tf.keras.layers.GlobalAvgPool2D()(features)
        # MoCoV2 paper adds a hidden layer
        dense = tf.keras.layers.Dense(num_hidden, activation="relu")(pooled)
        outpt = tf.keras.layers.Dense(output_dim)(dense)
        full_model = tf.keras.Model(inpt, outpt)
        
        #momentum_encoder = copy_model(full_model)
        momentum_encoder = tf.keras.models.clone_model(full_model)
        self._models = {"fcn":fcn, 
                        "full":full_model,
                        "momentum_encoder":momentum_encoder}
        
        # build training dataset
        self._ds = _build_augment_pair_dataset(trainingdata, 
                            imshape=imshape, batch_size=batch_size,
                            num_parallel_calls=num_parallel_calls, 
                            norm=norm, num_channels=num_channels, 
                            augment=augment, single_channel=single_channel)
        
        # create optimizer
        self._optimizer = self._build_optimizer(lr, lr_decay, opt_type="momentum",
                                                decay_type=decay_type)
        
        # build buffer
        K = batch_size*batches_in_buffer
        d = output_dim 
        self._buffer = tf.Variable(np.zeros((K,d), dtype=np.float32))
        
        # build training step
        self._training_step = build_momentum_contrast_training_step(
                full_model, 
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
                            alpha=alpha, tau=tau, output_dim=output_dim,
                            num_hidden=num_hidden,
                            lr=lr, lr_decay=lr_decay, 
                            imshape=imshape, num_channels=num_channels,
                            norm=norm, batch_size=batch_size,
                            num_parallel_calls=num_parallel_calls, 
                            single_channel=single_channel, notes=notes)
        self._prepopulate_buffer()
        
    def _prepopulate_buffer(self):
        i = 0
        bs = self.input_config["batch_size"]
        bib = self.config["batches_in_buffer"]
        #flatten = tf.keras.layers.GlobalAvgPool2D()
        for x,y in self._ds:
            k = tf.nn.l2_normalize(
                    self._models["momentum_encoder"](y, training=True), axis=1)
            _ = self._buffer[bs*i:bs*(i+1),:].assign(k)
            i += 1
            if i >= bib:
                break
        
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
                    hp.HParam("output_dim", hp.IntInterval(1, 1000000)):self.config["output_dim"],
                    hp.HParam("num_hidden", hp.IntInterval(1, 1000000)):self.config["num_hidden"]
                    }
            else:
                hparams=None
            self._linear_classification_test(hparams)
        
            
        
