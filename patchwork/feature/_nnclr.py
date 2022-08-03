import numpy as np
import tensorflow as tf

from patchwork.feature._generic import GenericExtractor, _TENSORBOARD_DESCRIPTIONS
from patchwork._util import compute_l2_loss, _compute_alignment_and_uniformity

from patchwork.feature._contrastive import _contrastive_loss, _build_augment_pair_dataset
from patchwork.feature._simclr import _gather, _build_embedding_model, SimCLRTrainer

try:
    bnorm = tf.keras.layers.experimental.SyncBatchNormalization
except:
    bnorm = tf.keras.layers.BatchNormalization




_DESCRIPTIONS = {
    "nt_xent_loss":"Contrastive crossentropy loss"
}
for d in _TENSORBOARD_DESCRIPTIONS:
    _DESCRIPTIONS[d] = _TENSORBOARD_DESCRIPTIONS[d]



def _find_nearest_neighbors(z,Q):
    """
    Find nearest neighbors for a batch of embeddings. Assumes
    that both z and Q are l2-normalized!!

    :z: (N,d) batch of embeddings
    :Q: (queue_size, d) support queue
    """
    similarity = tf.matmul(z,Q, transpose_b=True)
    # (N,)
    nn_indices = tf.argmax(similarity, axis=1)
    return tf.stop_gradient(tf.gather(Q, nn_indices))



def _update_queue(z,Q):
    """
    Update the support queue to include a new batch of embeddings
    :z: (N,d) tensor
    :Q: (queue_size, d) tf.Variable
    """
    queue_size = Q.shape[0]
    Q.assign(tf.concat([z,Q],0)[:queue_size,:])


def _initialize_queue(embed_model, ds, queue_size, num_channels=3):
    """
    Create and populate the support queue. Call within a scope
    context of a distribution strategy.
    """
    embeddings = []
    counter = 0
    for x, y in ds:
        embeddings.append(embed_model(x, training=True).numpy())
        counter += x.shape[0]
        if counter > queue_size:
            break

    embeddings = np.concatenate(embeddings, 0)[:queue_size, :]
    embeddings = tf.nn.l2_normalize(embeddings, 1)
    return tf.Variable(embeddings)


def _build_nnclr_training_step(embed_model, optimizer, Q, temperature=0.1,
                               weight_decay=0, eps=0):
    """
    Generate a tensorflow function to run the training step for NNCLR.

    :embed_model: full Keras model including both the convnet and
        projection head
    :optimizer: Keras optimizer
    :Q:
    :temperature: hyperparameter for scaling cosine similarities
    :weight_decay: coefficient for L2 loss
    :eps:

    The training function returns:
    :loss: value of the loss function for training
    :avg_cosine_sim: average value of the batch's matrix of dot products
    """
    # check whether we're in mixed-precision mode
    mixed = tf.keras.mixed_precision.global_policy().name == 'mixed_float16'

    def training_step(x, y):

        with tf.GradientTape() as tape:
            # run images through model and normalize embeddings
            z1 = tf.nn.l2_normalize(embed_model(x, training=True), 1)
            z2 = tf.nn.l2_normalize(embed_model(y, training=True), 1)

            nn1 = _find_nearest_neighbors(z1, Q)
            nn2 = _find_nearest_neighbors(z2, Q)

            # aggregate projections across replicas. z1 and z2 should
            # now correspond to the global batch size (gbs, d)
            z1 = _gather(z1)
            z2 = _gather(z2)
            nn1 = _gather(nn1)
            nn2 = _gather(nn2)

            xent_loss1, batch_acc1 = _contrastive_loss(z1, nn2, temperature, eps=eps)
            xent_loss2, batch_acc2 = _contrastive_loss(z2, nn1, temperature, eps=eps)

            xent_loss = 0.5 * (xent_loss1 + xent_loss2)
            batch_acc = 0.5 * (batch_acc1 + batch_acc2)

            if (weight_decay > 0) & ("LARS" not in optimizer._name):
                l2_loss = compute_l2_loss(embed_model)
            else:
                l2_loss = 0

            loss = xent_loss + weight_decay * l2_loss
            if mixed:
                loss = optimizer.get_scaled_loss(loss)

        gradients = tape.gradient(loss, embed_model.trainable_variables)
        if mixed:
            gradients = optimizer.get_unscaled_gradients(gradients)
        optimizer.apply_gradients(zip(gradients,
                                      embed_model.trainable_variables))

        # use one batch of embeddings to update the support queue
        _update_queue(z1, Q)

        return {"nt_xent_loss": xent_loss,
                "l2_loss": l2_loss,
                "loss": loss,
                "nce_batch_acc": batch_acc}

    return training_step


class NNCLRTrainer(SimCLRTrainer):
    """
    Class for training a NNCLR model.

    Based on "With a Little Help from My Friends: Nearest-Neighbor Contrastive
    Learning of Visual Representations" by Dwibedi et al.
    """
    modelname = "NNCLR"

    def __init__(self, logdir, trainingdata, testdata=None, fcn=None,
                 augment=True, temperature=0.1, num_hidden=2048, queue_size=32768,
                 output_dim=256, batchnorm=True, weight_decay=0,
                 num_projection_layers=3,
                 decoupled=False, eps=0, q=0,
                 lr=0.01, lr_decay=100000, decay_type="warmupcosine",
                 opt_type="adam",
                 imshape=(256,256), num_channels=3,
                 norm=255, batch_size=64, num_parallel_calls=None,
                 single_channel=False, notes="",
                 downstream_labels=None, strategy=None, jitcompile=False, **kwargs):
        """
        :logdir: (string) path to log directory
        :trainingdata: (list) list of paths to training images
        :testdata: (list) filepaths of a batch of images to use for eval
        :fcn: (keras Model) fully-convolutional network to train as feature extractor
        :augment: (dict) dictionary of augmentation parameters, True for defaults
        :temperature: (float) the Boltzmann temperature parameter- rescale the cosine similarities by this factor before computing softmax loss.
        :num_hidden: (int) number of hidden neurons in the network's projection head.
        :queue_size: (int) number of examples in the support queue
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
                                                weight_decay=weight_decay)

        # initialize the support queue
        with self.scope():
            self.Q = _initialize_queue(self._models["full"], ds,
                                   queue_size, num_channels=num_channels)
        # build training step
        step_fn = _build_nnclr_training_step(
                embed_model, self._optimizer, self.Q,
                temperature, weight_decay=weight_decay, eps=eps)
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
                            queue_size=queue_size,
                            weight_decay=weight_decay, batchnorm=batchnorm,
                            lr=lr, lr_decay=lr_decay,
                            decoupled=decoupled, eps=eps, q=q,
                            imshape=imshape, num_channels=num_channels,
                            norm=norm, batch_size=batch_size,
                            num_parallel_calls=num_parallel_calls,
                            single_channel=single_channel, notes=notes,
                            trainer="nnclr", strategy=str(strategy),
                            decay_type=decay_type, opt_type=opt_type, **kwargs)


