# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf

from patchwork.feature._generic import GenericExtractor
from patchwork._augment import augment_function
from patchwork.loaders import _image_file_dataset
from patchwork._util import compute_l2_loss
from patchwork.loaders import  _build_rotation_dataset



def _build_rotation_training_step(model, optimizer, weight_decay=0):
    """
    :model: keras model mapping image to 4-category classification
    :optimizer: keras optimizer
    :weight_decay: float; 0 to disable

    Returns a tf.function to use as a training step
    """

    @tf.function
    def training_step(x,y):

        with tf.GradientTape() as tape:
            z = model(x, training=True)
            loss = tf.reduce_mean(
                tf.keras.losses.sparse_categorical_crossentropy(y, z)
                )

            if weight_decay > 0:
                loss += weight_decay*compute_l2_loss(model)

        # update fast model
        variables = model.trainable_variables
        gradients = tape.gradient(loss, variables)
        optimizer.apply_gradients(zip(gradients, variables))
        return {"loss":loss}
    return training_step





class RotationTrainer(GenericExtractor):
    """
    Rotation-task trainer, based on "Unsupervised Representation Learning
    by Predicting Image Rotations" by Gidaris et al
    https://arxiv.org/abs/1803.07728
    """
    modelname = "Rotation"

    def __init__(self, logdir, trainingdata, testdata=None, fcn=None,
                 augment={"flip_left_right":True},
                 num_hidden=2048, weight_decay=0,
                 lr=0.01, lr_decay=100000, decay_type="exponential",
                 opt_type="momentum",
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
        :weight_decay: L2 loss weight; 0 to disable
        :lr: (float) initial learning rate
        :lr_decay: (int) number of steps for one decay period (0 to disable)
        :decay_type: (string) how to decay the learning rate- "exponential" (smooth exponential decay), "staircase" (non-smooth exponential decay), or "cosine"
        :opt_type: (str) which optimizer to use; "momentum" or "adam"
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
        self.logdir = logdir
        self.trainingdata = trainingdata
        self._downstream_labels = downstream_labels

        self._file_writer = tf.summary.create_file_writer(logdir, flush_millis=10000)
        self._file_writer.set_as_default()

        # if no FCN is passed- build one
        with self.scope():
            if fcn is None:
                fcn = tf.keras.applications.ResNet50V2(weights=None, include_top=False)
            self.fcn = fcn
            # make a prediction model
            inpt = tf.keras.layers.Input((imshape[0], imshape[1], num_channels))
            net = fcn(inpt)
            net =  tf.keras.layers.GlobalAvgPool2D()(net)
            net = tf.keras.layers.Dense(num_hidden, activation="relu")(net)
            net = tf.keras.layers.Dense(4, activation="softmax")(net)
            full_model = tf.keras.Model(inpt, net)
            self._models = {"fcn":fcn,
                        "full":full_model}

        # build training dataset
        self._ds = _build_rotation_dataset(trainingdata,
                            imshape=imshape, batch_size=batch_size,
                            num_parallel_calls=num_parallel_calls,
                            norm=norm, num_channels=num_channels,
                            augment=augment, single_channel=single_channel)

        # create optimizer
        self._optimizer = self._build_optimizer(lr, lr_decay, opt_type=opt_type,
                                                decay_type=decay_type)


        # build training step
        self._training_step = _build_rotation_training_step(
                full_model,
                self._optimizer, weight_decay)
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
                            num_hidden=num_hidden, weight_decay=weight_decay,
                            lr=lr, lr_decay=lr_decay, opt_type=opt_type,
                            imshape=imshape, num_channels=num_channels,
                            norm=norm, batch_size=batch_size,
                            num_parallel_calls=num_parallel_calls,
                            single_channel=single_channel, notes=notes,
                            trainer="rotation")


    def _run_training_epoch(self, **kwargs):
        """

        """
        for x, y in self._ds:
            lossdict = self._training_step(x,y)

            self._record_scalars(**lossdict)
            self._record_scalars(learning_rate=self._get_current_learning_rate())

            self.step += 1


    def evaluate(self, avpool=True):
        if self._downstream_labels is not None:
            self._linear_classification_test(avpool=avpool)





