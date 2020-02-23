# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf

from patchwork.feature._generic import GenericExtractor

from patchwork.feature._simclr import _build_simclr_dataset 
from patchwork.feature._simclr import _build_embedding_model
from patchwork.feature._simclr import _build_simclr_training_step


BIG_NUMBER = 1000.

def _build_distributed_training_step(strategy, embed_model, 
                                     optimizer, temperature=0.1):
    """
    
    """
    replicas = strategy.num_replicas_in_sync
    @tf.function
    def train_step(x, y):
        def step_fn(x,y):
            eye = tf.linalg.eye(x.shape[0])
            index = tf.range(0, x.shape[0])
            # the labels tell which similarity is the "correct" one- the augmented
            # pair from the same image. so index+y should look like [1,0,3,2,5,4...]
            labels = index+y

            with tf.GradientTape() as tape:
                # run each image through the convnet and
                # projection head
                embeddings = embed_model(x, training=True)
                # normalize the embeddings
                embeds_norm = tf.nn.l2_normalize(embeddings, axis=1)
                # compute the pairwise matrix of cosine similarities
                sim = tf.matmul(embeds_norm, embeds_norm, transpose_b=True)
                # subtract a large number from diagonals to effectively remove
                # them from the sum, and rescale by temperature
                logits = (sim - BIG_NUMBER*eye)/temperature
            
                loss = tf.reduce_mean(
                        tf.nn.sparse_softmax_cross_entropy_with_logits(labels, logits))/replicas

                gradients = tape.gradient(loss, embed_model.trainable_variables)
                optimizer.apply_gradients(zip(gradients,
                                      embed_model.trainable_variables))
                return loss
        
        per_example_losses = strategy.experimental_run_v2(
                                        step_fn, args=(x,y))
        print(per_example_losses)
        #mean_loss = strategy.reduce(
        #                tf.distribute.ReduceOp.MEAN, 
        #                per_example_losses, axis=0)
        #return mean_loss
        return per_example_losses
    return train_step



class DistributedSimCLRTrainer(GenericExtractor):
    """
    Class for training a SimCLR model.
    
    Based on "A Simple Framework for Contrastive Learning of Visual
    Representations" by Chen et al.
    """

    def __init__(self, strategy, logdir, trainingdata, testdata=None, fcn=None, 
                 augment=True, temperature=1., num_hidden=128,
                 output_dim=64,
                 lr=0.01, lr_decay=100000,
                 imshape=(256,256), num_channels=3,
                 norm=255, batch_size=64, num_parallel_calls=None,
                 single_channel=False, notes="",
                 downstream_labels=None):
        """
        :strategy: a tf.distribute Strategy object.
        :logdir: (string) path to log directory
        :trainingdata: (list) list of paths to training images
        :testdata: (list) filepaths of a batch of images to use for eval
        :fcn: (keras Model) fully-convolutional network to train as feature extractor
        :augment: (dict) dictionary of augmentation parameters, True for defaults
        :temperature: the Boltmann temperature parameter- rescale the cosine similarities by this factor before computing softmax loss.
        :num_hidden: number of hidden neurons in the network's projection head
        :output_dim: dimension of projection head's output space. Figure 8 in Chen et al's paper shows that their results did not depend strongly on this value.
        :lr: (float) initial learning rate
        :lr_decay: (int) steps for learning rate to decay by half (0 to disable)
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
        self._strategy = strategy
        self.logdir = logdir
        self.trainingdata = trainingdata
        self._downstream_labels = downstream_labels
        
        self._file_writer = tf.summary.create_file_writer(logdir, flush_millis=10000)
        self._file_writer.set_as_default()
        
        # if no FCN is passed- build one
        if fcn is None:
            with strategy.scope():
                fcn = tf.keras.applications.ResNet50V2(weights=None, include_top=False)
        self.fcn = fcn
        # Create a Keras model that wraps the base encoder and 
        # the projection head
        with strategy.scope():
            embed_model = _build_embedding_model(fcn, imshape, num_channels,
                                             num_hidden, output_dim)
        
        self._models = {"fcn":fcn, 
                        "full":embed_model}
        
        # build training dataset
        self._ds = strategy.experimental_distribute_dataset(
                _build_simclr_dataset(trainingdata, 
                                      imshape=imshape, batch_size=batch_size,
                                      num_parallel_calls=num_parallel_calls,
                                      norm=norm, num_channels=num_channels,
                                      augment=augment,
                                      single_channel=single_channel))
        
        # create optimizer
        if lr_decay > 0:
            learnrate = tf.keras.optimizers.schedules.ExponentialDecay(lr, 
                                            decay_steps=lr_decay, decay_rate=0.5,
                                            staircase=False)
        else:
            learnrate = lr
        with strategy.scope():
            self._optimizer = tf.keras.optimizers.Adam(learnrate)
        
        
        # build training step
        self._training_step = _build_distributed_training_step(strategy,
                                        embed_model, 
                                        self._optimizer, 
                                        temperature=temperature)
        
        self._test = False
        self._test_labels = None
        self._old_test_labels = None
        
        self.step = 0
        
        # parse and write out config YAML
        self._parse_configs(augment=augment, temperature=temperature,
                            num_hidden=num_hidden, output_dim=output_dim,
                            lr=lr, lr_decay=lr_decay, 
                            imshape=imshape, num_channels=num_channels,
                            norm=norm, batch_size=batch_size,
                            num_parallel_calls=num_parallel_calls, 
                            single_channel=single_channel, notes=notes)

    def _run_training_epoch(self, **kwargs):
        """
        
        """
        for x, y in self._ds:
            loss = self._training_step(x,y)
            
            self._record_scalars(loss=loss)
            self.step += 1
            
 
    def evaluate(self):
        if self._downstream_labels is not None:
            # choose the hyperparameters to record
            if not hasattr(self, "_hparams_config"):
                from tensorboard.plugins.hparams import api as hp
                hparams = {
                    hp.HParam("temperature", hp.RealInterval(0., 10000.)):self.config["temperature"],
                    hp.HParam("num_hidden", hp.IntInterval(1, 1000000)):self.config["num_hidden"],
                    hp.HParam("output_dim", hp.IntInterval(1, 1000000)):self.config["output_dim"]
                    }
            else:
                hparams=None
            self._linear_classification_test(hparams)
        
