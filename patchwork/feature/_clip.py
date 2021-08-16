# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
import sklearn.feature_extraction

from patchwork._util import compute_l2_loss
from patchwork.feature._generic import GenericExtractor
from patchwork.loaders import dataset


def _add_1d_residual_attention_block(inpt, num_heads, key_dim, attention_axes=[1]):
    """
    Add a residual attention block to a Keras model. patterned after this:
    
    https://github.com/openai/CLIP/blob/fa56f2525191a013533338f137aab59ac36d8c26/clip/model.py#L167
    """
    d_model = inpt.shape[-1]
    x = tf.keras.layers.LayerNormalization()(inpt)
    x = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim,
                                          attention_axes=attention_axes)(x,x)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.Dense(4*d_model)(x)
    x = tf.keras.layers.Activation(tf.nn.gelu)(x)
    x = tf.keras.layers.Dense(d_model)(x)
    return x


def build_bagofwords_transformer(d, project_to=512, num_layers=12, num_heads=12, key_dim=None, final_projection=False):
    """
    Build a simple transformer for text encoded as a bag-of-words. Project to a lower dimension, then
    add residual attention blocks and(optionally) a final linear projection/
    """
    if key_dim is None:
        key_dim = project_to
        
    inpt = tf.keras.layers.Input((d,))
    x = tf.keras.layers.Dense(project_to, activation="relu")(inpt)
    for n in range(num_layers):
        x = _add_1d_residual_attention_block(x, num_heads, key_dim)
    if final_projection:
        x = tf.keras.layers.Dense(final_projection)(x)
    
    return tf.keras.Model(inpt, x)


def get_vocab(corpus, max_features=5000, min_df=1, max_df=1., **kwargs):
    """
    """
    vec = sklearn.feature_extraction.text.CountVectorizer(max_features=max_features, min_df=min_df, 
                                                          max_df=max_df, **kwargs)
    vec.fit(corpus)
    vocab = list(vec.vocabulary_.keys())
    return vocab

def multihot_encode(x, depth):
    onehot = tf.one_hot(x, depth=depth)
    return tf.reduce_sum(onehot, 1)


def build_text_encoder(vocab, project_to=512, num_layers=12, num_heads=12, key_dim=None, final_projection=False):
    """
    
    """
    V = len(vocab)
    vectorizer = tf.keras.layers.experimental.preprocessing.TextVectorization(vocabulary=vocab)
    encoder = tf.keras.layers.Lambda(lambda x: multihot_encode(x, V))
    tfm = build_bagofwords_transformer(V, project_to, num_layers, num_heads, key_dim, final_projection)
    
    inpt = tf.keras.layers.Input((), dtype=tf.string)
    x = vectorizer(inpt)
    x = encoder(x)
    x = tfm(x)
    return tf.keras.Model(inpt, x)  



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

def compute_nce_loss(img_embed, text_embed, temp=0.07):
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
    return 0.5*(loss1 + loss2)




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



class CLIPTrainer(GenericExtractor):
    """
    Class for training a CLIP model. 
    
    Based on "Learning transferable visual models from natural language
    supervision" by Radford et al.
    """
    modelname = "CLIP"

    def __init__(self, logdir, vocab, trainingdata, traininglabels, 
                 testdata=None, fcn=None, 
                 augment=True, temperature=0.07, output_dim=64, 
                 project_to=512, num_layers=12, num_heads=12,
                 weight_decay=0,
                 lr=0.01, lr_decay=0, decay_type="cosine",
                 opt_type="adam",
                 imshape=(256,256), num_channels=3,
                 norm=255, batch_size=64, num_parallel_calls=None,
                 single_channel=False, notes="",
                 downstream_labels=None, stratify=None, strategy=None):
        """
        :logdir: (string) path to log directory
        :vocab:
        :trainingdata: (list) list of paths to training images
        :traininglabels:
        :testdata: (list) filepaths of a batch of images to use for eval
        :fcn: (keras Model) fully-convolutional network to train as feature extractor
        :augment: (dict) dictionary of augmentation parameters, True for defaults
        :temperature: the Boltzmann temperature parameter- rescale the cosine similarities by this factor before computing softmax loss.
        :output_dim: dimension of projection head's output space. 
        :project_to:
        :num_layers:
        :num_heads:
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
        
        # if no FCN is passed- build one
        with self.scope():
            if fcn is None:
                fcn = tf.keras.applications.ResNet50(weights=None, include_top=False)
            self.fcn = fcn
            # Create a Keras model that wraps the base encoder and 
            # the projection head
            full = build_image_encoder(fcn, num_channels=num_channels, 
                                       output_dim=output_dim)
            text = build_text_encoder(vocab, project_to=project_to, num_layers=num_layers,
                                      num_heads=num_heads, key_dim=project_to,
                                      final_projection=output_dim)
        
        self._models = {"fcn":fcn, 
                        "full":full,
                        "text":text}
        
        # build training dataset
        
        
        # we need to find the output size of the FCN
        mock_input = np.zeros((1,imshape[0], imshape[1], num_channels), dtype=np.float32)
        outshp = fcn(mock_input).shape # should be rank-4: (1,h,w,d)
        ds = dataset(trainingdata, traininglabels, imshape=imshape,
                     num_channels=num_channels, num_parallel_calls=num_parallel_calls,
                     norm=norm, batch_size=batch_size, shuffle=True)
        self._ds = self._distribute_dataset(ds)
        
        # create optimizer
        self._optimizer = self._build_optimizer(lr, lr_decay, opt_type=opt_type,
                                                decay_type=decay_type)
        
        
        # build training step
        self._training_step = build_clip_training_step(full, text, 
                                                       self._optimizer, temp=temperature,
                                                       weight_decay=weight_decay)
        """
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
        """
        self.step = 0
        
        # parse and write out config YAML
        self._parse_configs(augment=augment, temperature=temperature,
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
        for x1, seg1, x2, seg2 in self._ds:
            lossdict = self._training_step(x1, seg1, x2, seg2)
            self._record_scalars(**lossdict)
            self._record_scalars(learning_rate=self._get_current_learning_rate())
            self.step += 1
             
    def evaluate(self, avpool=True, query_fig=False):
        """
        if self._test:
            # if the user passed out-of-sample data to test- compute
            # alignment and uniformity measures
            alignment, uniformity = _compute_alignment_and_uniformity(
                                            self._test_ds, self._models["full"])
            
            self._record_scalars(alignment=alignment,
                             uniformity=uniformity, metric=True)
            metrics=["linear_classification_accuracy",
                                 "alignment",
                                 "uniformity"]
        else:
            metrics=["linear_classification_accuracy"]
        
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
        
