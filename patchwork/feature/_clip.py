# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
import logging
import os

from patchwork._util import compute_l2_loss
from patchwork.feature._generic import GenericExtractor
from patchwork.feature._text_transformer import build_text_transformer
from patchwork.loaders import _image_file_dataset
from patchwork._augment import augment_function


try:
    import sentencepiece as spm
except:
    logging.debug("unable to import sentencepiece- CLIPTrainer won't work.")


def clip_dataset(imfiles, labels, encoder, maxlen=76, imshape=(256,256), 
                 num_channels=3, num_parallel_calls=None, norm=255, 
                 batch_size=256, augment=False, shuffle=True,
                 single_channel=False):
    """
    
    """
    ds = _image_file_dataset(imfiles, ys=labels, imshape=imshape, 
                                        augment=augment, shuffle=shuffle)

    if augment:
        aug_func = augment_function(imshape, {"rot90":True})


    def _encode_text(y):
        y = str(y)
        y = encoder.encode(y, out_type=int, add_bos=True, add_eos=True)
        N = len(y)
        if N > maxlen:
            y = y[:maxlen]
        elif N < maxlen:
            y += [0]*(maxlen-N)
        return np.array(y)

    def _augment_and_encode(x,y):
        if augment:
            x = aug_func(x)
        y = tf.py_function(_encode_text, (y,), Tout=tf.int64)
        return x,y
    
    ds = ds.map(_augment_and_encode, num_parallel_calls=num_parallel_calls)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(1)
    return ds



def build_image_encoder(fcn, num_channels=3, output_dim=64):
    """
    NOT the full version used in OpenAI's paper- just a linear
    projection head after the global average pool, instead of
    a multi-head attention mechanism
    """
    inpt = tf.keras.layers.Input((None, None, 3))
    x = fcn(inpt)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(output_dim)(x)
    return tf.keras.Model(inpt, x)

def compute_nce_loss(img_embed, text_embed, temp=0.07, return_acc=False):
    """
    Symmetrized NCE loss for paired image/text embeddings
    """
    N = img_embed.shape[0]
    img_norm = tf.nn.l2_normalize(img_embed, 1)
    text_norm = tf.nn.l2_normalize(text_embed, 1)
    # NOTE this is different from what's described in the paper- check 
    # pseudocode in figure 3
    logits1 = tf.matmul(img_norm, text_norm, transpose_b=True)/temp
    labels1 = tf.range(N)
    loss1 = tf.reduce_mean(
        tf.losses.sparse_categorical_crossentropy(labels1, logits1, from_logits=True))
    
    logits2 = tf.matmul(text_norm, img_norm, transpose_b=True)/temp
    labels2 = tf.range(N)
    loss2 = tf.reduce_mean(
        tf.losses.sparse_categorical_crossentropy(labels2, logits2, from_logits=True))
    loss = 0.5*(loss1 + loss2)
    if return_acc:
        pred = tf.argmax(logits1, 1)
        acc = tf.reduce_mean(tf.cast(pred == labels1, tf.float32))
        return loss, acc
    
    return loss




def build_clip_training_step(img_model, text_model, optimizer, temp=0.07, weight_decay=0):
    trainvars = img_model.trainable_variables + text_model.trainable_variables
    def trainstep(img_batch, text_batch):
        with tf.GradientTape() as tape:
            img_embed = img_model(img_batch, training=True)
            text_embed = text_model(text_batch, training=True)
            
            nce_loss = compute_nce_loss(img_embed, text_embed, temp)
            if weight_decay > 0:
                l2_loss = compute_l2_loss(img_model) + compute_l2_loss(text_model)
            else:
                l2_loss = 0
                
            loss = nce_loss + weight_decay*l2_loss
        grads = tape.gradient(loss, trainvars)
        optimizer.apply_gradients(zip(grads, trainvars))
        lossdict = {"loss":loss, "l2_loss":l2_loss, "nce_loss":nce_loss}
        return lossdict
    return trainstep


def build_clip_test_step(img_model, text_model, optimizer, temp=0.07, weight_decay=0):
    @tf.function
    def teststep(img_batch, text_batch):
        img_embed = img_model(img_batch, training=False)
        text_embed = text_model(text_batch, training=False)
            
        nce_loss, acc = compute_nce_loss(img_embed, text_embed, temp, True)
        if weight_decay > 0:
            l2_loss = compute_l2_loss(img_model) + compute_l2_loss(text_model)
        else:
            l2_loss = 0
                
        loss = nce_loss + weight_decay*l2_loss
        return loss, acc
    return teststep



class CLIPTrainer(GenericExtractor):
    """
    Class for training a CLIP model. 
    
    Based on "Learning transferable visual models from natural language
    supervision" by Radford et al.
    """
    modelname = "CLIP"

    def __init__(self, logdir, tokenizer, trainingdata, traininglabels, 
                 testdata=None, testlabels=None, fcn=None,  augment=True,
                 maxlen=76, embed_dim=512, ff_dim=2048,
                 num_layers=12, num_heads=8,
                 temperature=0.07, output_dim=64, 
                 project_to=512,
                 weight_decay=0,
                 lr=0.01, lr_decay=0, decay_type="cosine",
                 opt_type="adam",
                 imshape=(256,256), num_channels=3,
                 norm=255, batch_size=64, num_parallel_calls=None,
                 single_channel=False, notes="",
                 downstream_labels=None, stratify=None, strategy=None):
        """
        :logdir: (string) path to log directory
        :tokenizer: (string) path to sentencepiece model file
        :trainingdata: (list) list of paths to training images
        :traininglabels:
        :testdata: (list) filepaths of a batch of images to use for eval
        :testlabels:
        :fcn: (keras Model) fully-convolutional network to train as feature extractor
        :augment: (dict) dictionary of augmentation parameters, True for defaults
        :maxlen:
        :embed_dim:
        :ff_dim:
        :num_layers:
        :num_heads:
        :temperature: the Boltzmann temperature parameter- rescale the cosine similarities by this factor before computing softmax loss.
        :output_dim: dimension of projection head's output space. 
        :project_to:
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
        self.logdir = logdir
        self.trainingdata = trainingdata
        self._downstream_labels = downstream_labels
        if strategy is None:
            strategy = tf.distribute.get_strategy()
        self.strategy = strategy
        
        self._file_writer = tf.summary.create_file_writer(logdir, flush_millis=10000)
        self._file_writer.set_as_default()
        # load tokenizer
        self._tokenizer = spm.SentencePieceProcessor(tokenizer)
        self._vocab_size = self._tokenizer._vocab_size()
        
        # if no FCN is passed- build one
        with self.scope():
            if fcn is None:
                fcn = tf.keras.applications.ResNet50(weights=None, include_top=False)
            self.fcn = fcn
            # Create a Keras model that wraps the base encoder and 
            # the projection head
            full = build_image_encoder(fcn, num_channels=num_channels, 
                                       output_dim=output_dim)
            text = build_text_transformer(self._vocab_size, maxlen,
                                          embed_dim=embed_dim, num_layers=num_layers,
                                          num_heads=num_heads, ff_dim=ff_dim,
                                          final_projection=output_dim)
        
        self._models = {"fcn":fcn, 
                        "full":full,
                        "text":text}
        
        # build training dataset
        ds = clip_dataset(trainingdata, traininglabels, self._tokenizer,
                          maxlen=maxlen, imshape=imshape, 
                          num_channels=num_channels, 
                          num_parallel_calls=num_parallel_calls,
                          norm=norm, batch_size=batch_size, shuffle=True)
        self._ds = self._distribute_dataset(ds)
        
        # create optimizer
        self._optimizer = self._build_optimizer(lr, lr_decay, opt_type=opt_type,
                                                decay_type=decay_type)
        
        
        # build training step
        self._training_step = build_clip_training_step(full, text, 
                                                       self._optimizer, temp=temperature,
                                                       weight_decay=weight_decay)
        
        if testdata is not None:
            self._test_ds = clip_dataset(testdata, testlabels, self._tokenizer,
                                         maxlen=maxlen, imshape=imshape,
                                         num_channels=num_channels,
                                         num_parallel_calls=num_parallel_calls,
                                         norm=norm, batch_size=batch_size, shuffle=False)
            @tf.function
            def loss_step(x,y):
                return compute_nce_loss(x, y, temp=temperature, return_acc=True)
            self._loss_step = loss_step
            self._test = True
        else:
            self._test = False
        
        self.step = 0
        
        # parse and write out config YAML
        self._parse_configs(tokenizer=tokenizer, maxlen=maxlen,
                            augment=augment, temperature=temperature,
                            output_dim=output_dim, weight_decay=weight_decay,
                            project_to=project_to, num_layers=num_layers, 
                            num_heads=num_heads,
                            lr=lr, lr_decay=lr_decay, 
                            imshape=imshape, num_channels=num_channels,
                            norm=norm, batch_size=batch_size,
                            num_parallel_calls=num_parallel_calls,
                            single_channel=single_channel, notes=notes,
                            trainer="clip", strategy=str(strategy),
                            decay_type=decay_type, opt_type=opt_type)

    def _run_training_epoch(self, **kwargs):
        """
        
        """
        for x, y in self._ds:
            lossdict = self._training_step(x, y)
            self._record_scalars(**lossdict)
            self._record_scalars(learning_rate=self._get_current_learning_rate())
            self.step += 1
             
    def evaluate(self, avpool=True, query_fig=False):
        
        if self._test:
            test_acc = []
            test_loss = []
            for x, y in self._test_ds:
                l, a = self._loss_step(x,y)
                test_acc.append(a.numpy())
                test_loss.append(l.numpy())
                
            self._record_scalars(test_acc=np.mean(test_acc),
                                 test_loss=np.mean(test_loss))
            # if the user passed out-of-sample data to test- compute
            # alignment and uniformity measures
            #alignment, uniformity = _compute_alignment_and_uniformity(
            #                                self._test_ds, self._models["full"])
            
            #self._record_scalars(alignment=alignment,
            #                 uniformity=uniformity, metric=True)
            #metrics=["linear_classification_accuracy",
            #                     "alignment",
            #                     "uniformity"]
        #else:
        #    metrics=["linear_classification_accuracy"]
        """
        if self._downstream_labels is not None:
            # choose the hyperparameters to record
            if not hasattr(self, "_hparams_config"):
                from tensorboard.plugins.hparams import api as hp
                hparams = {
                    hp.HParam("temperature", hp.RealInterval(0., 10000.)):self.config["temperature"],
                    hp.HParam("mean_scale", hp.RealInterval(0., 10000.)):self.config["mean_scale"],
                    hp.HParam("num_samples", hp.IntInterval(1, 1000000)):self.config["num_samples"],
                    hp.HParam("beta", hp.RealInterval(0., 10000.)):self.config["beta"],
                    hp.HParam("tau_plus", hp.RealInterval(0., 10000.)):self.config["tau_plus"],
                    hp.HParam("num_hidden", hp.IntInterval(1, 1000000)):self.config["num_hidden"],
                    hp.HParam("output_dim", hp.IntInterval(1, 1000000)):self.config["output_dim"],
                    hp.HParam("lr", hp.RealInterval(0., 10000.)):self.config["lr"],
                    hp.HParam("lr_decay", hp.RealInterval(0., 10000.)):self.config["lr_decay"],
                    hp.HParam("decay_type", hp.Discrete(["cosine", "exponential"])):self.config["decay_type"],
                    hp.HParam("weight_decay", hp.RealInterval(0., 10000.)):self.config["weight_decay"],
                    hp.HParam("batchnorm", hp.Discrete([True, False])):self.config["batchnorm"],
                    }
                for k in self.augment_config:
                    if isinstance(self.augment_config[k], float):
                        hparams[hp.HParam(k, hp.RealInterval(0., 10000.))] = self.augment_config[k]
            else:
                hparams=None

            self._linear_classification_test(hparams,
                        metrics=metrics, avpool=avpool, query_fig=query_fig)
            """
            
    def save(self):
        """
        Write model(s) to disk
        
        Note: tried to use SavedModel format for this and got a memory leak;
        think it's related to https://github.com/tensorflow/tensorflow/issues/32234
        
        For now sticking with HDF5
        """
        for m in self._models:
            path = os.path.join(self.logdir, m)
            self._models[m].save(path, overwrite=True, save_format="tf")
        
