# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from patchwork.feature._generic import GenericExtractor
from patchwork._augment import augment_function
from patchwork.loaders import _image_file_dataset
from patchwork._util import compute_l2_loss

BIG_NUMBER = 1000.



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
    if isinstance(imfiles, tf.data.Dataset) or isinstance(imfiles[0], str):
        _aug = augment_function(imshape, augment)
        @tf.function
        def _augment_and_stack(x):
            y = tf.constant(np.array([1,-1]).astype(np.int32))
            return tf.stack([_aug(x),_aug(x)]), y

        ds = ds.map(_augment_and_stack, num_parallel_calls=num_parallel_calls)
    # DUAL-INPUT CASE
    else:
        if isinstance(imshape[0], int): imshape = (imshape, imshape)
        _aug0 = augment_function(imshape[0], augment)
        _aug1 = augment_function(imshape[1], augment)
        @tf.function
        def _augment_and_stack(x0,x1):
            y = tf.constant(np.array([1,-1]).astype(np.int32))
            return (tf.stack([_aug0(x0),_aug0(x0)]), tf.stack([_aug1(x1),_aug1(x1)])), y
        ds = ds.map(_augment_and_stack, num_parallel_calls=num_parallel_calls)
        
    ds = ds.unbatch()
    ds = ds.batch(2*batch_size, drop_remainder=True)
    ds = ds.prefetch(1)
    return ds


def _build_embedding_model(fcn, imshape, num_channels, num_hidden, output_dim):
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
    net = tf.keras.layers.Dense(num_hidden, activation="relu")(net)
    net = tf.keras.layers.Dense(output_dim)(net)
    embedding_model = tf.keras.Model(inpt, net)
    return embedding_model





def _build_simclr_training_step(embed_model, optimizer, temperature=0.1,
                                weight_decay=0):
    """
    Generate a tensorflow function to run the training step for SimCLR.
    
    :embed_model: full Keras model including both the convnet and 
        projection head
    :optimizer: Keras optimizer
    :temperature: hyperparameter for scaling cosine similarities
    :weight_decay: coefficient for L2 loss
    
    The training function returns:
    :loss: value of the loss function for training
    :avg_cosine_sim: average value of the batch's matrix of dot products
    """
    # adding the tf.function decorator here causes errors when we
    # distribute across multiple GPUs
    #@tf.function 
    def training_step(x,y):
        lossdict = {}
        eye = tf.linalg.eye(y.shape[0])
        index = tf.range(0, y.shape[0])
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
                    tf.nn.sparse_softmax_cross_entropy_with_logits(labels, logits))
            
            # the original SimCLR paper used a weight decay of 1e-6
            if weight_decay > 0:
                lossdict["l2_loss"] = compute_l2_loss(embed_model)
                lossdict["nt_xent_loss"] = loss
                loss += weight_decay*lossdict["l2_loss"]
            lossdict["loss"] = loss

        gradients = tape.gradient(loss, embed_model.trainable_variables)
        optimizer.apply_gradients(zip(gradients,
                                      embed_model.trainable_variables))

        lossdict["avg_cosine_sim"] = tf.reduce_mean(sim)
        return lossdict
    return training_step


def random_adjust_aug_params(a, sigma=0.05):
    new_aug = {}
    for k in a:
        if isinstance(a[k], float):
            if k == "hue_delta":
                maxval = 0.5
            else:
                maxval = 1.
            new_aug[k] = round(min(max(a[k]+np.random.normal(0,sigma),0),
                                   maxval), 2)
        else:
            new_aug[k] = a[k]
    return new_aug


def find_new_aug_params(trainer, testfiles, num_trials=25, sigma=0.05):
    """
    EXPERIMENTAL
    
    Pass a trained SimCLRTrainer object and a list of test files, 
    and try to guess at a better set of augmentation parameters.
    
    The method will randomly jitter the existing parameters [num_trials]
    times with a standard deviation of [sigma], and return whichever
    set of params has the highest NCE loss on the test files.
    """
    aug_params = trainer.augment_config
    
    @tf.function
    def test_loss(x,y):
        eye = tf.linalg.eye(y.shape[0])
        index = tf.range(0, y.shape[0])
        labels = index+y

        embeddings = trainer._models["full"](x)
        embeds_norm = tf.nn.l2_normalize(embeddings, axis=1)
        sim = tf.matmul(embeds_norm, embeds_norm, transpose_b=True)
        logits = (sim - BIG_NUMBER*eye)/trainer.config["temperature"]
            
        loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(labels, logits))
        return loss
    
    results = []

    for i in tqdm(range(num_trials)):
        loss = 0
        aug = random_adjust_aug_params(aug_params, sigma)
        ds = _build_simclr_dataset(testfiles, augment=aug,
                                   **trainer.input_config)
        for x, y in ds:
            loss += test_loss(x,y).numpy()
        
        results.append((aug, loss))
    argmax = np.argmax([x[1] for x in results])
    return results[argmax][0], results


class SimCLRTrainer(GenericExtractor):
    """
    Class for training a SimCLR model.
    
    Based on "A Simple Framework for Contrastive Learning of Visual
    Representations" by Chen et al.
    """

    def __init__(self, logdir, trainingdata, testdata=None, fcn=None, 
                 augment=True, temperature=1., num_hidden=128,
                 output_dim=64, weight_decay=0,
                 lr=0.01, lr_decay=100000, decay_type="exponential",
                 opt_type="adam",
                 imshape=(256,256), num_channels=3,
                 norm=255, batch_size=64, num_parallel_calls=None,
                 single_channel=False, notes="",
                 downstream_labels=None, stratify=None, strategy=None):
        """
        :logdir: (string) path to log directory
        :trainingdata: (list) list of paths to training images
        :testdata: (list) filepaths of a batch of images to use for eval
        :fcn: (keras Model) fully-convolutional network to train as feature extractor
        :augment: (dict) dictionary of augmentation parameters, True for defaults
        :temperature: the Boltzmann temperature parameter- rescale the cosine similarities by this factor before computing softmax loss.
        :num_hidden: number of hidden neurons in the network's projection head
        :output_dim: dimension of projection head's output space. Figure 8 in Chen et al's paper shows that their results did not depend strongly on this value.
        :weight_decay: coefficient for L2-norm loss. The original SimCLR paper used 1e-6.
        :lr: (float) initial learning rate
        :lr_decay: (int) steps for learning rate to decay by half (0 to disable)
        :decay_type: (str) how to decay learning rate; "exponential" or "cosine"
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
        :stratify: pass a list of image labels here to stratify by batch
            during training
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
                                             num_hidden, output_dim)
        
        self._models = {"fcn":fcn, 
                        "full":embed_model}
        
        # build training dataset
        ds = _build_simclr_dataset(trainingdata, 
                                   imshape=imshape, batch_size=batch_size,
                                   num_parallel_calls=num_parallel_calls, 
                                   norm=norm, num_channels=num_channels, 
                                   augment=augment,
                                   single_channel=single_channel,
                                   stratify=stratify)
        self._ds = self._distribute_dataset(ds)
        
        # create optimizer
        self._optimizer = self._build_optimizer(lr, lr_decay, opt_type=opt_type,
                                                decay_type=decay_type)
        
        
        # build training step
        step_fn = _build_simclr_training_step(
                embed_model, self._optimizer, 
                temperature, weight_decay=weight_decay)
        self._training_step = self._distribute_training_function(step_fn)
        
        if testdata is not None:
            self._test_ds = _build_simclr_dataset(testdata, 
                                        imshape=imshape, batch_size=batch_size,
                                        num_parallel_calls=num_parallel_calls, 
                                        norm=norm, num_channels=num_channels, 
                                        augment=augment,
                                        single_channel=single_channel)
            
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
            self._test_loss = test_loss
            self._test = True
        else:
            self._test = False
        
        self.step = 0
        
        # parse and write out config YAML
        self._parse_configs(augment=augment, temperature=temperature,
                            num_hidden=num_hidden, output_dim=output_dim,
                            weight_decay=weight_decay,
                            lr=lr, lr_decay=lr_decay, 
                            imshape=imshape, num_channels=num_channels,
                            norm=norm, batch_size=batch_size,
                            num_parallel_calls=num_parallel_calls,
                            single_channel=single_channel, notes=notes,
                            trainer="simclr", strategy=str(strategy),
                            decay_type=decay_type)

    def _run_training_epoch(self, **kwargs):
        """
        
        """
        for x, y in self._ds:
            lossdict = self._training_step(x,y)
            self._record_scalars(**lossdict)
            self._record_scalars(learning_rate=self._get_current_learning_rate())
            self.step += 1
             
    def evaluate(self):
        if self._test:
            test_loss = 0
            for x,y in self._test_ds:
                loss, sim = self._test_loss(x,y)
                test_loss += loss.numpy()
                
            self._record_scalars(test_loss=test_loss)
            # I'm commenting out this tensorboard image- takes up a lot of
            # space but doesn't seem to add much
            #self._record_images(scalar_products=tf.expand_dims(tf.expand_dims(sim,-1), 0))
        if self._downstream_labels is not None:
            # choose the hyperparameters to record
            if not hasattr(self, "_hparams_config"):
                from tensorboard.plugins.hparams import api as hp
                hparams = {
                    hp.HParam("temperature", hp.RealInterval(0., 10000.)):self.config["temperature"],
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
        
