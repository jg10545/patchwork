# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
import tensorflow as tf

from patchwork.feature._generic import GenericExtractor, _TENSORBOARD_DESCRIPTIONS
from patchwork._util import compute_l2_loss, _compute_alignment_and_uniformity

from patchwork.feature._contrastive import _build_augment_pair_dataset

try:
    bnorm = tf.keras.layers.experimental.SyncBatchNormalization
except:
    bnorm = tf.keras.layers.BatchNormalization




_DESCRIPTIONS = {
    "mse_loss":"Mean-squared error loss between projections",
    "std_loss":"Standard deviation loss to prevent mode collapse. Standard deviation is computed across the batch for each dimension in feature space. Hinge loss is computed on 1-std(x).",
    "cov_loss":"Covariance loss computed across features"

}
for d in _TENSORBOARD_DESCRIPTIONS:
    _DESCRIPTIONS[d] = _TENSORBOARD_DESCRIPTIONS[d]



def _build_expander(fcn, imshape, num_channels, num_hidden, batchnorm=True):
    """
    Create a Keras model that wraps the base encoder and
    "expander" projection head
    """
    inpt = tf.keras.layers.Input((imshape[0], imshape[1], num_channels))
    net = fcn(inpt)
    net = tf.keras.layers.GlobalAvgPool2D(dtype="float32")(net)
    for i in range(2):
        net = tf.keras.layers.Dense(num_hidden, dtype="float32")(net)
        net = bnorm(dtype="float32")(net)
        net = tf.keras.layers.Activation("relu", dtype="float32")(net)
    net = tf.keras.layers.Dense(num_hidden, dtype="float32")(net)

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



def _cov_loss(x):
    N,d = x.shape
    # get dxd matrix of
    cov = tf.matmul(x, x, transpose_a=True)/(N-1)
    # subtract off the diagonal elements
    return (tf.reduce_sum(cov**2) - tf.reduce_sum(tf.linalg.diag_part(cov)**2))/d



def _build_vicreg_training_step(embed_model, optimizer, lam=25, mu=25, nu=1,
                                weight_decay=0, eps=1e-4):
    """
    Generate a tensorflow function to run the training step for VICReg.

    :embed_model: full Keras model including both the convnet and
        expander head
    :optimizer: Keras optimizer
    :lam: loss weight for __
    :mu: loss weight for __
    :nu: loss weight for __
    :weight_decay: coefficient for L2 loss

    The training function returns:
    :loss: value of the loss function for training
    """
    # check whether we're in mixed-precision mode
    mixed = tf.keras.mixed_precision.global_policy().name == 'mixed_float16'
    def training_step(x,y):


        with tf.GradientTape() as tape:
            # run images through model and normalize embeddings
            z1 = embed_model(x, training=True)
            z2 = embed_model(y, training=True)

            # MSE "invariance" loss- doesn't require gathered embeddings
            repr_loss = tf.reduce_mean(tf.losses.mse(z1,z2))

            # gather across GPUs
            z1 = _gather(z1)
            z2 = _gather(z2)

            batch_size, d = z1.shape

            # variance/coviariance losses
            z1 = z1 - tf.reduce_mean(z1, axis=0, keepdims=True)
            z2 = z2 - tf.reduce_mean(z2, axis=0, keepdims=True)
            # variance
            std_1 = tf.math.sqrt(tf.math.reduce_variance(z1, axis=0)+eps)
            std_2 = tf.math.sqrt(tf.math.reduce_variance(z2, axis=0)+eps)
            std_loss = tf.reduce_mean(tf.nn.relu(1-std_1) + tf.nn.relu(1-std_2) )/2
            # covariance
            cov_loss = _cov_loss(z1) + _cov_loss(z2)

            if (weight_decay > 0)&("LARS" not in optimizer._name):
                l2_loss = compute_l2_loss(embed_model)
            else:
                l2_loss = 0

            loss = lam*repr_loss + mu*std_loss + nu*cov_loss + weight_decay*l2_loss

            if mixed:
                loss = optimizer.get_scaled_loss(loss)

        gradients = tape.gradient(loss, embed_model.trainable_variables)
        if mixed:
            gradients = optimizer.get_unscaled_gradients(gradients)
        optimizer.apply_gradients(zip(gradients,
                                      embed_model.trainable_variables))


        return {"std_loss":std_loss,
                "l2_loss":l2_loss,
                "loss":loss,
                "cov_loss":cov_loss,
                "mse_loss":repr_loss}
    return training_step



class VICRegTrainer(GenericExtractor):
    """
    Class for training a VICReg model.

    Based on "VICREG: VARIANCE-INVARIANCE-COVARIANCE RE- GULARIZATION FOR SELF-SUPERVISED LEARNING" by Bardes et al.
    """
    modelname = "VICReg"

    def __init__(self, logdir, trainingdata, testdata=None, fcn=None,
                 augment=True, lam=25, mu=25, nu=1, num_hidden=8192,
                 weight_decay=0, lr=0.01, lr_decay=100000,
                 decay_type="exponential", opt_type="adam",
                 imshape=(256,256), num_channels=3,
                 norm=255, batch_size=64, num_parallel_calls=None,
                 single_channel=False, notes="",
                 downstream_labels=None, strategy=None, jitcompile=False):
        """
        :logdir: (string) path to log directory
        :trainingdata: (list) list of paths to training images
        :testdata: (list) filepaths of a batch of images to use for eval
        :fcn: (keras Model) fully-convolutional network to train as feature extractor
        :augment: (dict) dictionary of augmentation parameters, True for defaults
        :lam: coefficient for MSE loss
        :mu: coefficient for variance loss
        :nu: coefficient for covariance loss
        :num_hidden: number of hidden neurons in the network's projection head.
        :weight_decay: coefficient for L2-norm loss.
        :lr: (float) initial learning rate
        :lr_decay:  (int) number of steps for one decay period (0 to disable)
        :decay_type: (string) how to decay the learning rate- "exponential" (smooth exponential decay), "staircase" (non-smooth exponential decay), "cosine", or "warmupcosine"
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
            embed_model = _build_expander(fcn, imshape, num_channels,
                                             num_hidden)

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
                                                weight_decay=weight_decay)


        # build training step
        step_fn = _build_vicreg_training_step(
                embed_model, self._optimizer,
                lam=lam, mu=mu, nu=nu,
                weight_decay=weight_decay)
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
        self._parse_configs(augment=augment, lam=lam, mu=mu, nu=nu,
                            num_hidden=num_hidden,
                            weight_decay=weight_decay,
                            lr=lr, lr_decay=lr_decay,
                            imshape=imshape, num_channels=num_channels,
                            norm=norm, batch_size=batch_size,
                            num_parallel_calls=num_parallel_calls,
                            single_channel=single_channel, notes=notes,
                            trainer="vicreg", strategy=str(strategy),
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



