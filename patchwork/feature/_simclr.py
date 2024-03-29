# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf

from patchwork.feature._generic import GenericExtractor, _TENSORBOARD_DESCRIPTIONS
from patchwork._util import compute_l2_loss, _compute_alignment_and_uniformity

from patchwork.feature._contrastive import _contrastive_loss, _build_augment_pair_dataset

try:
    bnorm = tf.keras.layers.experimental.SyncBatchNormalization
except:
    bnorm = tf.keras.layers.BatchNormalization




_DESCRIPTIONS = {
    "nt_xent_loss":"Contrastive crossentropy loss"
}
for d in _TENSORBOARD_DESCRIPTIONS:
    _DESCRIPTIONS[d] = _TENSORBOARD_DESCRIPTIONS[d]



def _build_embedding_model(fcn, imshape, num_channels, num_hidden, output_dim,
                           batchnorm=True, num_projection_layers=2):
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
    net = tf.keras.layers.GlobalAvgPool2D(dtype="float32")(net)
    # NORMAL SimCLR CASE
    if num_hidden > 0:
        # for every projection layer except the final one: ReLU activation
        for _ in range(num_projection_layers-1):
            net = tf.keras.layers.Dense(num_hidden, dtype="float32")(net)
            if batchnorm == "layernorm":
                net = tf.keras.layers.LayerNormalization(dtype="float32")(net)
            elif batchnorm:
                net = bnorm(dtype="float32")(net)
            net = tf.keras.layers.Activation("relu", dtype="float32")(net)
        # for the final layer- linear activation and no bias
        net = tf.keras.layers.Dense(output_dim, use_bias=False, dtype="float32")(net)
        if bool(batchnorm)&(batchnorm != "layernorm"):
            net = bnorm(dtype="float32")(net)

    else:
        # DirectCLR CASE
        net = tf.keras.layers.Lambda(lambda x: x[:,:output_dim])(net)
    embedding_model = tf.keras.Model(inpt, net)
    return embedding_model



def _gather(tensor):
    context = tf.distribute.get_replica_context()
    rep_id = context.replica_id_in_sync_group

    strategy =  tf.distribute.get_strategy()
    num_replicas = strategy.num_replicas_in_sync

    if num_replicas < 2:
        return tensor

    ext_tensor = tf.scatter_nd(
        indices=[[rep_id]],
        updates=[tensor],
        shape=tf.concat([[num_replicas], tf.shape(tensor)], axis=0))

    ext_tensor = context.all_reduce(tf.distribute.ReduceOp.SUM,
                                            ext_tensor)
    return tf.reshape(ext_tensor, [-1] + ext_tensor.shape.as_list()[2:])

def _build_simclr_training_step(embed_model, optimizer, temperature=0.1,
                                weight_decay=0, decoupled=False, eps=0,
                                q=0):
    """
    Generate a tensorflow function to run the training step for SimCLR.

    :embed_model: full Keras model including both the convnet and
        projection head
    :optimizer: Keras optimizer
    :temperature: hyperparameter for scaling cosine similarities
    :weight_decay: coefficient for L2 loss
    :decoupled:
    :eps:
    :data_parallel:

    The training function returns:
    :loss: value of the loss function for training
    :avg_cosine_sim: average value of the batch's matrix of dot products
    """
    # check whether we're in mixed-precision mode
    mixed = tf.keras.mixed_precision.global_policy().name == 'mixed_float16'
    def training_step(x,y):

        with tf.GradientTape() as tape:
            # run images through model and normalize embeddings
            z1 = tf.nn.l2_normalize(embed_model(x, training=True), 1)
            z2 = tf.nn.l2_normalize(embed_model(y, training=True), 1)

            # aggregate projections across replicas. z1 and z2 should
            # now correspond to the global batch size (gbs, d)
            z1 = _gather(z1)
            z2 = _gather(z2)


            xent_loss, batch_acc = _contrastive_loss(z1, z2, temperature,
                                          decoupled, eps=eps, q=q)


            if (weight_decay > 0)&("LARS" not in optimizer._name):
                l2_loss = compute_l2_loss(embed_model)
            else:
                l2_loss = 0

            loss = xent_loss + weight_decay*l2_loss
            if mixed:
                loss = optimizer.get_scaled_loss(loss)

        gradients = tape.gradient(loss, embed_model.trainable_variables)
        if mixed:
            gradients = optimizer.get_unscaled_gradients(gradients)
        optimizer.apply_gradients(zip(gradients,
                                      embed_model.trainable_variables))


        return {"nt_xent_loss":xent_loss,
                "l2_loss":l2_loss,
                "loss":loss,
                "nce_batch_acc":batch_acc}
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




class SimCLRTrainer(GenericExtractor):
    """
    Class for training a SimCLR model.

    Based on "A Simple Framework for Contrastive Learning of Visual
    Representations" by Chen et al.
    """
    modelname = "SimCLR"

    def __init__(self, logdir, trainingdata, testdata=None, fcn=None,
                 augment=True, temperature=0.1, num_hidden=2048,
                 output_dim=2048, batchnorm=True, weight_decay=0,
                 num_projection_layers=2,
                 decoupled=False, eps=0, q=0,
                 lr=0.01, lr_decay=100000, decay_type="exponential",
                 warmup_steps=5000,
                 opt_type="adam",
                 imshape=(256,256), num_channels=3,
                 norm=255, batch_size=64, num_parallel_calls=None,
                 single_channel=False, notes="",
                 downstream_labels=None, strategy=None, jitcompile=False,
                 initial_step=0, **kwargs):
        """
        :logdir: (string) path to log directory
        :trainingdata: (list) list of paths to training images
        :testdata: (list) filepaths of a batch of images to use for eval
        :fcn: (keras Model) fully-convolutional network to train as feature extractor
        :augment: (dict) dictionary of augmentation parameters, True for defaults
        :temperature: (float) the Gibbs temperature parameter- rescale the cosine similarities by this factor before computing softmax loss.
        :num_hidden: (int) number of hidden neurons in the network's projection head. Set to 0 to
            use the DirectCLR method from "Understanding Dimesnional Collapse in Contrastive
            Self-Supervised Learning"
        :output_dim: (int) dimension of projection head's output space. Figure 8 in Chen et al's paper shows that their results did not depend strongly on this value.
        :batchnorm: (bool) whether to include batch normalization in the projection head.
        :weight_decay: (float) coefficient for L2-norm loss. The original SimCLR paper used 1e-6.
        :num_projection_layers: (int) number of layers in the projection head, including the output layer
        :decoupled: (bool) if True, use the modified loss function from "Decoupled Contrastive
            Learning" by Yeh et al
        :eps: (float) epsilon parameter from the Implicit Feature Modification paper ("Can
             contrastive learning avoid shortcut solutions?" by Robinson et al)
        :q: (float) RINCE loss parameter from "Robust Contrastive Learning against Noisy Views"
            by Chuang et al
        :lr: (float) initial learning rate
        :lr_decay:  (int) number of steps for one decay period (0 to disable)
        :decay_type: (string) how to decay the learning rate- "exponential" (smooth exponential decay), "staircase" (non-smooth exponential decay), "cosine", or "warmupcosine"
        :warmup_steps: (int) number of steps for warmupcosine decay type to ramp up over
        :opt_type: (string) optimizer type; "adam", "momentum", or "lars"
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
        :initial_step: start optimizer at this step
        """
        assert augment is not False, "this method needs an augmentation scheme"
        self.logdir = logdir
        self.trainingdata = trainingdata
        self._downstream_labels = downstream_labels
        self.strategy = strategy
        self._description = _DESCRIPTIONS

        self._file_writer = tf.summary.create_file_writer(logdir, flush_millis=10000)
        self._file_writer.set_as_default()

        # if no FCN is passed- build one
        with self.scope():
            if fcn is None:
                fcn = tf.keras.applications.ResNet50(weights=None, include_top=False)
            self.fcn = fcn
            # Create a Keras model that wraps the base encoder and
            # the projection head
            embed_model = _build_embedding_model(fcn, imshape, num_channels,
                                             num_hidden, output_dim, batchnorm,
                                             num_projection_layers)

        self._models = {"fcn":fcn,
                        "full":embed_model}

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
                                                decay_type=decay_type,
                                                weight_decay=weight_decay,
                                                initial_step=initial_step,
                                                warmup_steps=warmup_steps)


        # build training step
        step_fn = _build_simclr_training_step(
                embed_model, self._optimizer,
                temperature, weight_decay=weight_decay,
                decoupled=decoupled, eps=eps, q=q)
        self._training_step = self._distribute_training_function(step_fn,
                                                                 jitcompile=jitcompile)

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
        self._parse_configs(augment=augment, temperature=temperature,
                            num_hidden=num_hidden, output_dim=output_dim,
                            weight_decay=weight_decay, batchnorm=batchnorm,
                            lr=lr, lr_decay=lr_decay,
                            decoupled=decoupled, eps=eps, q=q,
                            imshape=imshape, num_channels=num_channels,
                            norm=norm, batch_size=batch_size,
                            num_parallel_calls=num_parallel_calls,
                            single_channel=single_channel, notes=notes,
                            trainer="simclr", strategy=str(strategy),
                            decay_type=decay_type, opt_type=opt_type, **kwargs)

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


