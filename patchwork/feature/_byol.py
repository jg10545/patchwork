# -*- coding: utf-8 -*-
"""

    Implementation of BYOL from "Bootstrap Your Own Latent: A New Approach to 
    Self-Supervised Learning" by Grill et al.


"""
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from patchwork.feature._generic import GenericExtractor
from patchwork.feature._moco import exponential_model_update, _build_augment_pair_dataset

from patchwork._augment import augment_function
from patchwork.loaders import _image_file_dataset
from patchwork._util import compute_l2_loss, _compute_alignment_and_uniformity


_DESCRIPTIONS = {
    "online_target_cosine_similarity":"""
    Average cosine similarity between all the weights of the target network and corresponding weights of the online network. 
    """,
    "test_proj_avg_cosine_sim":"""
    Average pairwise cosine similarity between projections of (augmented) images in the test set. If this converges toward1 it could mean you're getting mode collapse.'
    """,
    "mse_loss":"""
    Mean-squared error between online network prediction and target network projection for the training batch
    """}
    

def _perceptron(input_dim, num_hidden=4096, output_dim=256, batchnorm=True):
    # macro to build a single-hidden-layer perceptron
    inpt = tf.keras.layers.Input((input_dim,))
    net = tf.keras.layers.Dense(num_hidden)(inpt)
    if batchnorm:
        net = tf.keras.layers.BatchNormalization()(net)
    net = tf.keras.layers.Activation("relu")(net)
    net = tf.keras.layers.Dense(output_dim)(net)
    return tf.keras.Model(inpt, net)

def _build_models(fcn, imshape, num_channels, num_hidden=4096, output_dim=256):
    """
    Initialize all the model components we'll need for training
    """
    #  --- ONLINE MODEL ---
    # input view
    inpt = tf.keras.layers.Input((imshape[0], imshape[1], num_channels))
    # representation
    rep_tensor = fcn(inpt)
    rep = tf.keras.layers.Flatten()(rep_tensor)
    rep_dimension = rep.shape[-1]
    # projection
    online_projector = _perceptron(rep_dimension, num_hidden, output_dim)
    proj = online_projector(rep)
    online = tf.keras.Model(inpt, proj)
    # prediction
    online_predictor = _perceptron(output_dim, num_hidden, output_dim)
    #pred = online_predictor(proj)
    
    #  --- TARGET MODEL ---
    target = tf.keras.models.clone_model(online)
    
    return {"fcn":fcn, "online":online, "prediction":online_predictor,
            "target":target}


def _build_byol_dataset(imfiles, imshape=(256,256), batch_size=256, 
                      num_parallel_calls=None, norm=255,
                      num_channels=3, augment=True,
                      single_channel=False):
    """
    :stratify: if not None, a list of categories for each element in
        imfile.
    """
    assert augment != False, "don't you need to augment your data?"
    
    ds = _image_file_dataset(imfiles, imshape=imshape, 
                             num_parallel_calls=num_parallel_calls,
                             norm=norm, num_channels=num_channels,
                             shuffle=True, single_channel=single_channel,
                             augment=False)  
    
    _aug = augment_function(imshape, augment)
    @tf.function
    def pair_augment(x):
        return (_aug(x), _aug(x)), np.array([1])
    
    ds = ds.map(pair_augment, num_parallel_calls=num_parallel_calls)
    
    ds = ds.batch(batch_size)
    ds = ds.prefetch(1)
    return ds


def _mse_what(y_true, y_pred):
    # mean-squared error loss wrapper
    return y_true.shape[-1]*tf.reduce_mean(
        tf.keras.losses.mean_squared_error(y_true, y_pred))

def _mse(y_true, y_pred):
    # mean-squared error loss wrapper
    return tf.reduce_mean(
        tf.keras.losses.mean_squared_error(y_true, y_pred))


def _dot_product_loss(a,b):
    a = tf.nn.l2_normalize(a, axis=1)
    b = tf.nn.l2_normalize(b, axis=1)
    prod = tf.matmul(a, b, transpose_b=True)
    return 2-2*tf.reduce_mean(prod)

def _build_byol_training_step(online, prediction, target, optimizer,
                              tau, weight_decay=0):
    """
    
    """
    trainvars = online.trainable_variables + prediction.trainable_variables
    
    def training_step(x,y):
        x1, x2 = x
        lossdict = {}
        
        # target projections
        targ1 = tf.nn.l2_normalize(target(x1, training=False), 1)
        targ2 = tf.nn.l2_normalize(target(x2, training=False), 1)
        
        with tf.GradientTape() as tape:
            # online projections
            z1 = online(x1, training=True)
            z2 = online(x2, training=True)
            # online predictions
            pred1 = tf.nn.l2_normalize(prediction(z1, training=True), 1)
            pred2 = tf.nn.l2_normalize(prediction(z2, training=True), 1)
            # compute mean-squared error both ways
            mse_loss = _mse(targ1, pred2) + _mse(targ2, pred1)
            #mse_loss = _dot_product_loss(targ1, pred2) + _dot_product_loss(targ2, pred1)
            lossdict["loss"] = mse_loss
            lossdict["mse_loss"] = mse_loss

            if weight_decay > 0:
                lossdict["l2_loss"] = compute_l2_loss(online) + \
                            compute_l2_loss(prediction)
                lossdict["loss"] += weight_decay*lossdict["l2_loss"]
           
        # UPDATE WEIGHTS OF ONLINE MODEL
        gradients = tape.gradient(lossdict["loss"], trainvars)
        optimizer.apply_gradients(zip(gradients, trainvars))
        # UPDATE WEIGHTS OF TARGET MODEL
        #lossdict["target_online_avg_weight_diff"] = exponential_model_update(target, online, tau)
        
        return lossdict
    return training_step


def _build_target_bn_update_step(target):
    def update_step(x,y):
        x1, x2 = x
        t1 = target(x1, training=True)
        t2 = target(x2, training=True)
        
        av_mean_sq = 0.5*(tf.reduce_mean(t1**2)+tf.reduce_mean(t2**2))
        return {"av_sq_mean_from_update_step":av_mean_sq}
    return update_step
    

        
def _compare_model_weights(online, target):
    """
    Macro to build a summary stat to help us
    understand whether the online and target models
    are collapsing toward each other.
    
    Computes the average cosine similarity across
    all corresponding model weights
    """
    N = len(online.trainable_variables)
    cosine_sim = 0
    
    for o,t in zip(online.trainable_variables, target.trainable_variables):
        o = tf.nn.l2_normalize(tf.reshape(o, [-1]))
        t = tf.nn.l2_normalize(tf.reshape(t, [-1]))
        cosine_sim += tf.reduce_sum(o*t)
        
    return cosine_sim/N

def _average_cosine_sim(x):
    """
    Input a 2D tensor and return the average pairwise
    cosine similarity of rows in the tensor.
    """
    x_norm = tf.nn.l2_normalize(x, 1)
    return tf.reduce_mean(tf.matmul(x_norm, x_norm,
                                    transpose_b=True))
        




class BYOLTrainer(GenericExtractor):
    """
    Class for training a BYOL model.
    
    Based on  "Bootstrap Your Own Latent: A New Approach to 
    Self-Supervised Learning" by Grill et al.
    """

    def __init__(self, logdir, trainingdata, testdata=None, fcn=None, 
                 augment=True, num_hidden=4096, output_dim=256,
                 tau=0.996, alpha=0.99,  weight_decay=0, opt_type="momentum",
                 lr=0.01, lr_decay=100000, decay_type="exponential",
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
        :num_hidden: number of hidden neurons in the projection and prediction heads
        :output_dim: output dimension of projection and prediction heads
        :tau:
        :alpha:
        :weight_decay: coefficient for L2-norm loss. The original SimCLR paper used 1e-6.
        :opt_type:
        :lr: (float) initial learning rate
        :lr_decay: (int) steps for learning rate to decay by half (0 to disable)
        :decay_type: (str) how to decay learning rate; "exponential" or "cosine"
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
            Strategy object here. NOT YET TESTED
        """
        assert augment is not False, "this method needs an augmentation scheme"
        self.logdir = logdir
        self.trainingdata = trainingdata
        self._downstream_labels = downstream_labels
        self.strategy = strategy
        self._descriptions = _DESCRIPTIONS
        
        self._file_writer = tf.summary.create_file_writer(logdir, flush_millis=10000)
        self._file_writer.set_as_default()
        
        # if no FCN is passed- build one
        with self.scope():
            if fcn is None:
                fcn = tf.keras.applications.ResNet50V2(weights=None, include_top=False)
            self.fcn = fcn
            # Create Keras models for the full online model, online predictions,
            # and target model
            self._models = _build_models(fcn, imshape, num_channels,
                                         num_hidden, output_dim)
        self.update_batchnorm_momentum(alpha)
        
        # build training dataset
        ds = _build_byol_dataset(trainingdata, 
                                   imshape=imshape, batch_size=batch_size,
                                   num_parallel_calls=num_parallel_calls, 
                                   norm=norm, num_channels=num_channels, 
                                   augment=augment,
                                   single_channel=single_channel)
        self._ds = self._distribute_dataset(ds)
        
        # create optimizer
        self._optimizer = self._build_optimizer(lr, lr_decay, opt_type,
                                                decay_type=decay_type)
        
        
        # build training step
        step_fn = _build_byol_training_step(
                                    self._models["online"],
                                    self._models["prediction"],
                                    self._models["target"],
                                    self._optimizer,
                                    tau, weight_decay)
        self._training_step = self._distribute_training_function(step_fn)
        
        # weight update step - initially had this inside the training step, but
        # it gave some issues with multigpu training since I hadn't explicitly set
        # tf.VariableAggregation and tf.VariableSynchronization for each variable
        # in the network.
        @tf.function
        def weight_update_step():
            return exponential_model_update(self._models["target"], self._models["online"],
                                            self.config["tau"])
        self._weight_update_step = weight_update_step
        
        # THIS IS NOT THE MOST EFFICIENT WAY TO IMPLEMENT THIS
        # in the momentum^2 paper they subclass the pytorch batchnorm layer
        # to do what they want- in the name of easy compatibility, for now, 
        # we'll just run each batch through the target model in training mode
        # once (so the standard exponential udpate happens) in one step,
        # then run them through again in inference mode during the training step
        bn_update_step = _build_target_bn_update_step(self._models["target"])
        self._bn_update_step = self._distribute_training_function(bn_update_step)

        
        if testdata is not None:
            
            self._test_ds = _build_augment_pair_dataset(testdata, 
                                        imshape=imshape, batch_size=batch_size,
                                        num_parallel_calls=num_parallel_calls, 
                                        norm=norm, num_channels=num_channels, 
                                        augment=augment,
                                        single_channel=single_channel)
            @tf.function
            def cmw():
                return _compare_model_weights(self._models["online"], 
                                              self._models["target"])
            self._compare_model_weights = cmw

            self._test = True
        else:
            self._test = False
        
        self.step = 0
        
        # parse and write out config YAML
        self._parse_configs(augment=augment,
                            num_hidden=num_hidden, output_dim=output_dim,
                            weight_decay=weight_decay, tau=tau,
                            lr=lr, lr_decay=lr_decay, opt_type=opt_type,
                            imshape=imshape, num_channels=num_channels,
                            norm=norm, batch_size=batch_size,
                            num_parallel_calls=num_parallel_calls,
                            single_channel=single_channel, notes=notes,
                            trainer="byol", strategy=str(strategy),
                            decay_type=decay_type)

    def update_batchnorm_momentum(self, alpha):
        with self.scope():
            for layer in self._models["target"].layers:
                if hasattr(layer, "layers"):
                    for l in layer.layers:
                        if l.name.endswith("_bn"):
                            l.momentum = alpha


    def _run_training_epoch(self, **kwargs):
        """
        
        """
        for x, y in self._ds:
            # update target batchnorm running averages
            avg_sqs = self._bn_update_step(x,y)
            self._record_scalars(**avg_sqs)
            # train the online model
            lossdict = self._training_step(x,y)
            self._record_scalars(**lossdict)
            # update the target model
            self._record_scalars(target_online_avg_weight_diff=self._weight_update_step())
            
            self.step += 1
             
    def evaluate(self):
        if self._test:
            # compute average cosine similarity of target and online networks
            self._record_scalars(online_target_cosine_similarity=self._compare_model_weights())
            #
            proj = self._models["online"].predict(self._test_ds)
            acs = _average_cosine_sim(proj)
            self._record_scalars(test_proj_avg_cosine_sim=acs)
            """
            for x,y in self._test_ds:
                loss, sim = self._test_loss(x,y)
                test_loss += loss.numpy()
                
            self._record_scalars(test_loss=test_loss)
            # I'm commenting out this tensorboard image- takes up a lot of
            # space but doesn't seem to add much
            #self._record_images(scalar_products=tf.expand_dims(tf.expand_dims(sim,-1), 0))
            """
            # if the user passed out-of-sample data to test- compute
            # alignment and uniformity measures
            alignment, uniformity = _compute_alignment_and_uniformity(
                                            self._test_ds, self._models["fcn"])
            
            self._record_scalars(alignment=alignment,
                             uniformity=uniformity, metric=True)
            
        if self._downstream_labels is not None:
            # choose the hyperparameters to record
            if not hasattr(self, "_hparams_config"):
                from tensorboard.plugins.hparams import api as hp
                hparams = {
                    hp.HParam("tau", hp.RealInterval(0., 1.)):self.config["tau"],
                    hp.HParam("num_hidden", hp.IntInterval(1, 1000000)):self.config["num_hidden"],
                    hp.HParam("output_dim", hp.IntInterval(1, 1000000)):self.config["output_dim"],
                    hp.HParam("lr", hp.RealInterval(0., 10000.)):self.config["lr"],
                    hp.HParam("lr_decay", hp.RealInterval(0., 10000.)):self.config["lr_decay"],
                    hp.HParam("decay_type", hp.Discrete(["cosine", "exponential"])):self.config["decay_type"],
                    hp.HParam("weight_decay", hp.RealInterval(0., 10000.)):self.config["weight_decay"]
                    }
                for k in self.augment_config:
                    if isinstance(self.augment_config[k], float):
                        hparams[hp.HParam(k, hp.RealInterval(0., 10000.))] = self.augment_config[k]
            else:
                hparams=None
            self._linear_classification_test(hparams)
        
