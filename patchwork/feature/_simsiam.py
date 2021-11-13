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


def _build_embedding_models(fcn, imshape, num_channels, num_hidden=2048, pred_dim=512):
    """
    Create Keras models for the projection and prediction heads
    
    """
    # PROJECTION MODEL
    inpt = tf.keras.layers.Input((imshape[0], imshape[1], num_channels))
    net = fcn(inpt)
    net = tf.keras.layers.Flatten()(net)
    for i in range(3):
        net = tf.keras.layers.Dense(num_hidden, use_bias=False)(net)
        net = tf.keras.layers.BatchNormalization()(net)
        if i < 2:
            net = tf.keras.layers.Activation("relu")(net)
            
    projection = tf.keras.Model(inpt, net)
    
    # PREDICTION MODEL
    inpt = tf.keras.layers.Input((num_hidden))
    net = tf.keras.layers.Dense(pred_dim, use_bias=False)(inpt)
    net = tf.keras.layers.BatchNormalization()(net)
    net  = tf.keras.layers.Activation("relu")(net)
    net = tf.keras.layers.Dense(num_hidden)(net)
    prediction = tf.keras.Model(inpt, net)
    
    return projection, prediction


def _simsiam_loss(p,z):
    """
    
    """
    z = tf.stop_gradient(z)
    
    z = tf.nn.l2_normalize(z,1)
    p = tf.nn.l2_normalize(p,1)
    
    return tf.reduce_mean(
        tf.reduce_sum(-1*p*z, axis=1)
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

        gradients = tape.gradient(loss, embed_model.trainable_variables)
        optimizer.apply_gradients(zip(gradients,
                                      embed_model.trainable_variables))

        
        return {"simsiam_loss":ss_loss,
                "l2_loss":l2_loss,
                "loss":loss}
    return training_step




#-------------------------------------- NEW CODE HERE --------------------


def _build_simclr_dataset(imfiles, imshape=(256,256), batch_size=256, 
                      num_parallel_calls=None, norm=255,
                      num_channels=3, augment=True,
                      single_channel=False, stratify=None):
    """
    :stratify: if not None, a list of categories for each element in
        imfile.
    """
    
    if stratify is not None:
        categories = list(set(stratify))
        # SINGLE-INPUT CASE
        if isinstance(imfiles[0], str):
            file_lists = [[imfiles[i] for i in range(len(imfiles)) 
                        if stratify[i] == c]
                for c in categories
                ]
        # DUAL-INPUT
        else:
            file_lists = [([imfiles[0][i] for i in range(len(imfiles[0])) 
                        if stratify[i] == c],
                            [imfiles[1][i] for i in range(len(imfiles[1])) 
                        if stratify[i] == c] ) for c in categories ]
        datasets = [_build_simclr_dataset(f, imshape=imshape, 
                                          batch_size=batch_size, 
                                          num_parallel_calls=num_parallel_calls, 
                                          norm=norm, num_channels=num_channels, 
                                          augment=augment, 
                                          single_channel=single_channel,
                                          stratify=None)
                    for f in file_lists]
        return tf.data.experimental.sample_from_datasets(datasets)
    
    assert augment != False, "don't you need to augment your data?"
    
    
    ds = _image_file_dataset(imfiles, imshape=imshape, 
                             num_parallel_calls=num_parallel_calls,
                             norm=norm, num_channels=num_channels,
                             shuffle=True, single_channel=single_channel,
                             augment=False)  
    
    # SINGLE-INPUT CASE (DEFAULT)
    #if isinstance(imfiles, tf.data.Dataset) or isinstance(imfiles[0], str):
    _aug = augment_function(imshape, augment)
    @tf.function
    def _augment_and_stack(*x):
        # if there's only one input, augment it twice (standard SimCLR).
        # if there are two, augment them separately (case where user is
        # trying to express some specific semantics)
        x0 = tf.reshape(x[0], (imshape[0], imshape[1], num_channels))
        if len(x) == 2:
            x1 = tf.reshape(x[1], (imshape[0], imshape[1], num_channels))
        else:
            x1 = x0
        y = tf.constant(np.array([1,-1]).astype(np.int32))
        return tf.stack([_aug(x0),_aug(x1)]), y

    ds = ds.map(_augment_and_stack, num_parallel_calls=num_parallel_calls)
        
    ds = ds.unbatch()
    ds = ds.batch(2*batch_size, drop_remainder=True)
    ds = ds.prefetch(1)
    return ds


def _build_embedding_model(fcn, imshape, num_channels, num_hidden, output_dim, batchnorm=True):
    """
    Create a Keras model that wraps the base encoder and 
    the projection head
    """
    # SINGLE-INPUT CASE
    if len(fcn.inputs) == 1:
        inpt = tf.keras.layers.Input((imshape[0], imshape[1], num_channels))
    # DUAL-INPUT CASE
    else:
        if isinstance(imshape[0], int): imshape = (imshape, imshape)
        if isinstance(num_channels, int): num_channels=(num_channels, num_channels)
        inpt0 = tf.keras.layers.Input((imshape[0][0], imshape[0][1], 
                                       num_channels[0]))
        inpt1 = tf.keras.layers.Input((imshape[1][0], imshape[1][1], 
                                       num_channels[1]))
        inpt = [inpt0, inpt1]
    net = fcn(inpt)
    net = tf.keras.layers.Flatten()(net)
    net = tf.keras.layers.Dense(num_hidden)(net)
    if batchnorm:
        net = tf.keras.layers.BatchNormalization()(net)
    net = tf.keras.layers.Activation("relu")(net)
    net = tf.keras.layers.Dense(output_dim, use_bias=False)(net)
    embedding_model = tf.keras.Model(inpt, net)
    return embedding_model





class SimCLRTrainer(GenericExtractor):
    """
    Class for training a SimCLR model.
    
    Based on "A Simple Framework for Contrastive Learning of Visual
    Representations" by Chen et al.
    """
    modelname = "SimCLR"

    def __init__(self, logdir, trainingdata, testdata=None, fcn=None, 
                 augment=True, temperature=0.1, num_hidden=128,
                 output_dim=64, batchnorm=True, weight_decay=0,
                 decoupled=False, data_parallel=True,
                 lr=0.01, lr_decay=100000, decay_type="exponential",
                 opt_type="adam",
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
        :output_dim: dimension of projection head's output space. Figure 8 in Chen et al's paper shows that their results did not depend strongly on this value.
        :batchnorm: whether to include batch normalization in the projection head.
        :weight_decay: coefficient for L2-norm loss. The original SimCLR paper used 1e-6.
        :decoupled: if True, use the modified loss function from "Decoupled Contrastive 
            Learning" by Yeh et al
        :data_parallel: if True, compute contrastive loss only using negatives from
            within each replica
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
                fcn = tf.keras.applications.ResNet50V2(weights=None, include_top=False)
            self.fcn = fcn
            # Create a Keras model that wraps the base encoder and 
            # the projection head
            embed_model = _build_embedding_model(fcn, imshape, num_channels,
                                             num_hidden, output_dim, batchnorm)
        
        self._models = {"fcn":fcn, 
                        "full":embed_model}
        
        # build training dataset
        #ds = _build_simclr_dataset(trainingdata, 
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
        step_fn = _build_simclr_training_step(
                embed_model, self._optimizer, 
                temperature, weight_decay=weight_decay,
                decoupled=decoupled, data_parallel=data_parallel)
        self._training_step = self._distribute_training_function(step_fn)
        
        if testdata is not None:
            self._test_ds =  _build_augment_pair_dataset(testdata, 
                                        imshape=imshape, batch_size=batch_size,
                                        num_parallel_calls=num_parallel_calls, 
                                        norm=norm, num_channels=num_channels, 
                                        augment=augment,
                                        single_channel=single_channel)
            """
            @tf.function
            def test_loss(x,y):
                eye = tf.linalg.eye(y.shape[0])
                index = tf.range(0, y.shape[0])
                labels = index+y

                embeddings = self._models["full"](x)
                embeds_norm = tf.nn.l2_normalize(embeddings, axis=1)
                sim = tf.matmul(embeds_norm, embeds_norm, transpose_b=True)
                logits = (sim - BIG_NUMBER*eye)/self.config["temperature"]
            
                loss = tf.reduce_mean(
                    tf.nn.sparse_softmax_cross_entropy_with_logits(labels, logits))
                return loss, sim
            self._test_loss = test_loss"""
            self._test = True
        else:
            self._test = False
        
        self.step = 0
        
        # parse and write out config YAML
        self._parse_configs(augment=augment, temperature=temperature,
                            num_hidden=num_hidden, output_dim=output_dim,
                            weight_decay=weight_decay, batchnorm=batchnorm,
                            lr=lr, lr_decay=lr_decay, 
                            decoupled=decoupled, data_parallel=data_parallel,
                            imshape=imshape, num_channels=num_channels,
                            norm=norm, batch_size=batch_size,
                            num_parallel_calls=num_parallel_calls,
                            single_channel=single_channel, notes=notes,
                            trainer="simclr", strategy=str(strategy),
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
                                            self._test_ds, self._models["full"])
            
            self._record_scalars(alignment=alignment,
                             uniformity=uniformity, metric=True)
         
        if self._downstream_labels is not None:
            self._linear_classification_test(avpool=avpool,
                        query_fig=query_fig)
        
        

