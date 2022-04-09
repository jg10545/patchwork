# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf

from patchwork.feature._generic import GenericExtractor
from patchwork._augment import augment_function
from patchwork.loaders import _image_file_dataset
from patchwork._util import compute_l2_loss, _compute_alignment_and_uniformity

from patchwork.feature._moco import _build_augment_pair_dataset
from patchwork.feature._contrastive import _contrastive_loss

try:
    batchnorm = tf.keras.layers.experimental.SyncBatchNormalization
except:
    batchnorm = tf.keras.layers.BatchNormalization


def _build_embedding_models(fcn, imshape, num_channels, num_hidden=2048, pred_dim=512):
    """
    Create Keras models for the projection and prediction heads
    
    Returns full projection model (fcn+head) and prediction model
    """
    # PROJECTION MODEL f; z = f(x)
    inpt = tf.keras.layers.Input((imshape[0], imshape[1], num_channels))
    net = fcn(inpt)
    net = tf.keras.layers.GlobalAvgPool2D(dtype="float32")(net)
    # first layer
    #net = tf.keras.layers.Dense(num_hidden, use_bias=False)(net)
    #net = tf.keras.layers.BatchNormalization()(net)
    #net = tf.keras.layers.Activation("relu")(net)
    # second layer
    #net = tf.keras.layers.Dense(num_hidden, use_bias=False)(net)
    #net = tf.keras.layers.BatchNormalization()(net)
    #net = tf.keras.layers.Activation("relu")(net)
    
    for i in range(3):
        net = tf.keras.layers.Dense(num_hidden, use_bias=False, dtype="float32")(net)
        net = batchnorm(dtype="float32")(net)
        if i < 2:
            net = tf.keras.layers.Activation("relu", dtype="float32")(net)
            
    projection = tf.keras.Model(inpt, net)
    
    # PREDICTION MODEL h; p = h(z)
    inpt = tf.keras.layers.Input((num_hidden))
    net = tf.keras.layers.Dense(pred_dim, use_bias=False, dtype="float32")(inpt)
    net = batchnorm(dtype="float32")(net)
    net  = tf.keras.layers.Activation("relu", dtype="float32")(net)
    net = tf.keras.layers.Dense(num_hidden, dtype="float32")(net)
    
    prediction = tf.keras.Model(inpt, net)
    
    return projection, prediction


def _simsiam_loss(p,z):
    """
    Normalize embeddings, stop gradients through z 
    and compute dot product
    """
    z = tf.stop_gradient(tf.nn.l2_normalize(z,1))
    
    p = tf.nn.l2_normalize(p,1)
    
    return -1*tf.reduce_mean(
        tf.reduce_sum(p*z, axis=1)
        )


def _build_simsiam_training_step(embed_model, predict_model, optimizer, 
                                 weight_decay=0):
    """
    Generate a tensorflow function to run the training step for SimSiam
    
    :embed_model: full Keras model including both the convnet and 
        projection head
    :predict_model: prediction MLP
    :optimizer: Keras optimizer
    :weight_decay: coefficient for L2 loss
    
    The training function returns:
    :loss: value of the loss function for training
    """
    trainvars = embed_model.trainable_variables + predict_model.trainable_variables
    def training_step(x,y):
        
        with tf.GradientTape() as tape:
            # run images through model and normalize embeddings
            z1 = embed_model(x, training=True)
            z2 = embed_model(y, training=True)
            
            p1 = predict_model(z1, training=True)
            p2 = predict_model(z2, training=True)
            
            ss_loss = 0.5*_simsiam_loss(p1, z2) + 0.5*_simsiam_loss(p2, z1)
        
            if weight_decay > 0:
                l2_loss = compute_l2_loss(embed_model)
            else:
                l2_loss = 0
                
            loss = ss_loss + weight_decay*l2_loss

        gradients = tape.gradient(loss, trainvars)
        optimizer.apply_gradients(zip(gradients, trainvars))

        # let's check for output collapse- compute the standard deviation of
        # normalized embeddings along each direction in feature space. the average
        # should be close to 1/sqrt(d)
        d = z1.shape[-1]
        output_std = tf.reduce_mean(
                    tf.math.reduce_std(tf.nn.l2_normalize(z1,1),0))*np.sqrt(d)

        return {"simsiam_loss":ss_loss,
                "l2_loss":l2_loss,
                "loss":loss,
                "output_std":output_std}
    return training_step





class SimSiamTrainer(GenericExtractor):
    """
    Class for training a SimSiam model.
    
    Based on "Exploring Simple Siamese Representation Learning" by Chen and He
    """
    modelname = "SimSiam"

    def __init__(self, logdir, trainingdata, testdata=None, fcn=None, 
                 augment=True, num_hidden=2048, pred_dim=512,
                 weight_decay=0, lr=0.01, lr_decay=100000, 
                 decay_type="cosine", opt_type="momentum",
                 imshape=(256,256), num_channels=3,
                 norm=255, batch_size=64, num_parallel_calls=None,
                 single_channel=False, notes="",
                 downstream_labels=None, strategy=None):
        """
        :logdir: (string) path to log directory
        :trainingdata: (list) list of paths to training images
        :testdata: (list) filepaths of a batch of images to use for eval
        :fcn: (keras Model) fully-convolutional network to train as feature extractor
        :augment: (dict) dictionary of augmentation parameters, True for defaults
        :temperature: the Boltzmann temperature parameter- rescale the cosine similarities by this factor before computing softmax loss.
        :num_hidden: number of hidden neurons in the network's projection head
        :pred_dim:
        :weight_decay: coefficient for L2-norm loss. The original SimCLR paper used 1e-6.
        :lr: (float) initial learning rate
        :lr_decay:  (int) number of steps for one decay period (0 to disable)
        :decay_type: (string) how to decay the learning rate- "exponential" (smooth exponential decay), "staircase" (non-smooth exponential decay), or "cosine"
        :opt_type: (string) optimizer type; "adam" or "momentum"
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
        :strategy: if distributing across multiple GPUs, pass a tf.distribute
            Strategy object here
        """
        assert augment is not False, "this method needs an augmentation scheme"
        self.logdir = logdir
        self.trainingdata = trainingdata
        self._downstream_labels = downstream_labels
        self.strategy = strategy
        
        self._file_writer = tf.summary.create_file_writer(logdir, flush_millis=10000)
        self._file_writer.set_as_default()
        
        # if no FCN is passed- build one
        with self.scope():
            if fcn is None:
                fcn = tf.keras.applications.ResNet50(weights=None, include_top=False)
            self.fcn = fcn
            # Create a Keras model that wraps the base encoder and 
            # the projection head
            project, predict = _build_embedding_models(fcn, imshape, num_channels,
                                             num_hidden, pred_dim)
        
        self._models = {"fcn":fcn, 
                        "full":project,
                        "predict":predict}
        
        # build training dataset
        ds = _build_augment_pair_dataset(trainingdata,
                                   imshape=imshape, batch_size=batch_size,
                                   num_parallel_calls=num_parallel_calls, 
                                   norm=norm, num_channels=num_channels, 
                                   augment=augment,
                                   single_channel=single_channel)
        self._ds = self._distribute_dataset(ds)
        
        # create optimizer
        self._optimizer = self._build_optimizer(lr, lr_decay, opt_type=opt_type,
                                                decay_type=decay_type)
        
        
        # build training step
        step_fn = _build_simsiam_training_step(project, predict, 
                                               self._optimizer,
                                               weight_decay=weight_decay)
        self._training_step = self._distribute_training_function(step_fn)
        
        if testdata is not None:
            self._test_ds =  _build_augment_pair_dataset(testdata, 
                                        imshape=imshape, batch_size=batch_size,
                                        num_parallel_calls=num_parallel_calls, 
                                        norm=norm, num_channels=num_channels, 
                                        augment=augment,
                                        single_channel=single_channel)

            self._test = True
        else:
            self._test = False
        
        self.step = 0
        
        # parse and write out config YAML
        self._parse_configs(augment=augment,
                            num_hidden=num_hidden, pred_dim=pred_dim,
                            weight_decay=weight_decay, 
                            lr=lr, lr_decay=lr_decay, 
                            imshape=imshape, num_channels=num_channels,
                            norm=norm, batch_size=batch_size,
                            num_parallel_calls=num_parallel_calls,
                            single_channel=single_channel, notes=notes,
                            trainer="simsiam", strategy=str(strategy),
                            decay_type=decay_type, opt_type=opt_type)

    def _run_training_epoch(self, **kwargs):
        """
        
        """
        for x, y in self._ds:
            lossdict = self._training_step(x,y)
            self._record_scalars(**lossdict)
            self._record_scalars(learning_rate=self._get_current_learning_rate())
            self.step += 1
             
    def evaluate(self, avpool=True, query_fig=False):
        
        if self._test:
            # if the user passed out-of-sample data to test- compute
            # alignment and uniformity measures
            alignment, uniformity = _compute_alignment_and_uniformity(
                                            self._test_ds, self._models["fcn"])
            
            self._record_scalars(alignment=alignment,
                             uniformity=uniformity, metric=True)
         
        if self._downstream_labels is not None:
            self._linear_classification_test(avpool=avpool,
                        query_fig=query_fig)
        
        

