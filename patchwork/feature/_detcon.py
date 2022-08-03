# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf


from patchwork._util import compute_l2_loss, _compute_alignment_and_uniformity
from patchwork._augment import augment_function
from patchwork.feature._contrastive import _contrastive_loss

from patchwork.loaders import _image_file_dataset

from patchwork.feature._generic import GenericExtractor
from patchwork.feature._contrastive import _build_augment_pair_dataset
from patchwork.feature._detcon_utils import _get_segments, _get_grid_segments
from patchwork.feature._detcon_utils import _segment_aug, _prepare_mask
from patchwork.feature._detcon_utils import _prepare_embeddings, SEG_AUG_FUNCTIONS
from patchwork._augment import DEFAULT_SIMCLR_PARAMS


def _build_segment_pair_dataset(imfiles, mean_scale=1000, num_samples=16, outputsize=None,
                                imshape=(256,256), batch_size=256,
                      num_parallel_calls=None, norm=255,
                      num_channels=3, augment=True,
                      single_channel=False):
    """
    Build a tf.data.Dataset object for training pairs of augmented images with
    segmentation masks for each.

    :mean_scale: average scale parameter to use for Felzenszwalb's segmentation
        algorithm. if 0, segment using a 4x4 grid instead
    :num_samples: number of segments to sample from Felzenszwalb segmentation output
    :outputsize: (H,W) dimensions of output of FCN
    """
    assert augment, "don't you need to augment your data?"
    if augment == True:
        augment = DEFAULT_SIMCLR_PARAMS
    # build the initial image loader. Dataset yields unbatched images
    ds = _image_file_dataset(imfiles, imshape=imshape,
                             num_parallel_calls=num_parallel_calls,
                             norm=norm, num_channels=num_channels,
                             shuffle=True, single_channel=single_channel)
    # generate segments- using Felzenszwalb
    if mean_scale > 0:
        def _segment(x):
            _seg = lambda y: _get_segments(y, mean_scale=mean_scale, num_samples=num_samples)
            segments = tf.py_function(func=_seg, inp=(x,), Tout=(tf.float32))
            segments.set_shape((x.shape[0], x.shape[1], num_samples))
            return x, segments
    # or using a grid
    else:
        seg = _get_grid_segments(imshape, num_samples=num_samples)
        def _segment(x):
            return x, seg

    # dataset yields unbatched (image, segmentation) tuples
    ds = ds.map(_segment, num_parallel_calls=num_parallel_calls)
    # augment image and segmentation together
    def _seg_aug(img, seg):
        img1, aug1 = _segment_aug(img, seg, augment, imshape=imshape, outputsize=outputsize)
        img2, aug2 = _segment_aug(img, seg, augment, imshape=imshape, outputsize=outputsize)
        return img1, aug1, img2, aug2
    # dataset yields unbatched (image, segmentation, image, segmentation) tuples
    ds = ds.map(_seg_aug, num_parallel_calls=num_parallel_calls)
    # filter out cases where the previous augmentation step creates any empty segments
    #ds = ds.filter(_filter_out_bad_segments)
    # finally, augment images separately
    aug2 = {k:augment[k] for k in augment if k not in SEG_AUG_FUNCTIONS}
    _aug = augment_function(imshape, num_channels=num_channels, params=aug2)
    if len(aug2) > 0:
        def _augment_pair(img1, seg1, img2, seg2):
            #img1, seg1, img2, seg2 = x
            return _aug(img1), seg1, _aug(img2), seg2
        # dataset yields unbatches (image, segmentation, image, segmentation) tuples
        ds = ds.map(_augment_pair, num_parallel_calls=num_parallel_calls)

    ds = ds.batch(batch_size, drop_remainder=True)
    if num_parallel_calls == tf.data.AUTOTUNE:
        ds = ds.prefetch(tf.data.AUTOTUNE)
    else:
        ds = ds.prefetch(1)
    return ds





def _build_trainstep(fcn, projector, optimizer, strategy, temp=1, weight_decay=0,
                     dataparallel=False):
    """
    Build a distributed training step for SimCLR or HCL.

    Set tau_plus and beta to 0 for SimCLR parameters.

    :model: Keras projection model
    :optimizer: Keras optimizer
    :strategy: tf.distribute.Strategy object
    :temp: temperature parameter
    :weightdecay: L2 loss coefficient. 0 to disable
    :dataparallel: if True, compute contrastive loss separately on each replica
        without shuffling data between them

    Returns a distributed training function
    """
    # check whether we're in mixed-precision mode
    mixed = tf.keras.mixed_precision.global_policy().name == 'mixed_float16'
    trainvars = fcn.trainable_variables + projector.trainable_variables
    def _step(x1, m1, x2, m2):
        with tf.GradientTape() as tape:
            loss = 0

            #print("x,y:", x.shape, y.shape)

            # run images through model and normalize embeddings. do this
            # in three steps:
            # 1) compute features with FCN (N, w, h, feature_dim)
            # 2) compute segment-weighted features (N*num_samples, feature_dim)
            # 3) compute projections z (N*num_samples, d)
            x1 = fcn(x1, training=True)
            hm1 = _prepare_embeddings(x1, m1)
            z1 = tf.nn.l2_normalize(projector(hm1, training=True), 1)

            x2 = fcn(x2, training=True)
            hm2 = _prepare_embeddings(x2, m2)
            z2 = tf.nn.l2_normalize(projector(hm2, training=True), 1)

            # mask out all positive pairs where one mask or the other
            # is empty
            mask = tf.stop_gradient(_prepare_mask(m1, m2))

            # aggregate projections across replicas. z1 and z2 should
            # now correspond to the global batch size (gbs*num_samples, d)
            if not dataparallel:
                # get replica context- we'll use this to aggregate embeddings
                # across different GPUs
                context = tf.distribute.get_replica_context()
                z1 = context.all_gather(z1, 0)
                z2 = context.all_gather(z2, 0)
                mask = tf.stop_gradient(context.all_gather(mask))

            softmax_loss, nce_batch_acc = _contrastive_loss(z1*mask, z2*mask, temp)
            # SimCLR loss case
            #if (tau_plus == 0)&(beta == 0):
            #    softmax_prob, nce_batch_acc = _simclr_softmax_prob(z1, z2, temp, negmask)
            # HCL loss case
            #elif (tau_plus > 0)&(beta > 0):
            #    softmax_prob, nce_batch_acc = _hcl_softmax_prob(z1, z2, temp,
            #                                                    beta, tau_plus, negmask)
            #else:
            #    assert False, "both tau_plus and beta must be nonzero to run HCL"

            #softmax_loss = tf.reduce_mean(-1*mask*tf.math.log(softmax_prob))
            loss += softmax_loss

            if (weight_decay > 0)&("LARS" not in optimizer._name):
                l2_loss = compute_l2_loss(fcn) + compute_l2_loss(projector)
                loss += weight_decay*l2_loss
            else:
                l2_loss = 0

            if mixed:
                loss = optimizer.get_scaled_loss(loss)


        grad = tape.gradient(loss, trainvars)
        if mixed:
            grad = optimizer.get_unscaled_gradients(grad)
        optimizer.apply_gradients(zip(grad, trainvars))
        return {"loss":loss, "nt_xent_loss":softmax_loss,
                "l2_loss":l2_loss,
               "nce_batch_acc":nce_batch_acc}

    @tf.function
    def trainstep(x1, m1, x2, m2):
        per_example_losses = strategy.run(_step, args=(x1, m1, x2, m2))
        lossdict = {k:strategy.reduce(
                    tf.distribute.ReduceOp.MEAN,
                    per_example_losses[k], axis=None)
                    for k in per_example_losses}#

        return lossdict
    return trainstep
    #return _step


class DetConTrainer(GenericExtractor):
    """
    Class for training a DetCon_S model.

    Based on "Efficient Visual Pretraining With Contrastive Detection" by Henaff et al.

    https://arxiv.org/abs/2103.10957
    """
    modelname = "DetCon"

    def __init__(self, logdir, trainingdata, testdata=None, fcn=None,
                 augment=True, temperature=0.1, mean_scale=1000, num_samples=16,
                 dataparallel=False,
                 num_hidden=2048, output_dim=2048, batchnorm=True,
                 weight_decay=1e-6,
                 lr=0.01, lr_decay=0, decay_type="exponential",
                 opt_type="adam",
                 imshape=(256,256), num_channels=3,
                 norm=255, batch_size=64, num_parallel_calls=None,
                 single_channel=False, notes="",
                 downstream_labels=None, strategy=None, **kwargs):
        """
        :logdir: (string) path to log directory
        :trainingdata: (list) list of paths to training images
        :testdata: (list) filepaths of a batch of images to use for eval
        :fcn: (keras Model) fully-convolutional network to train as feature extractor
        :augment: (dict) dictionary of augmentation parameters, True for defaults
        :temperature: the Boltzmann temperature parameter- rescale the cosine similarities by this factor before computing softmax loss.
        :mean_scale: average scale parameter to use for Felzenszwalb's segmentation
        algorithm. if 0, segment using a 4x4 grid instead
        :num_samples: number of segments to sample from Felzenszwalb segmentation output
        :dataparallel: if True, compute loss function separately on each replica instead of
            combining negative examples acros replicas.
        :num_hidden: number of hidden neurons in the network's projection head
        :output_dim: dimension of projection head's output space.
        :batchnorm: whether to include batch normalization in the projection head.
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

        if dataparallel:
            bnorm = tf.keras.layers.BatchNormalization
        else:
            bnorm = tf.keras.layers.experimental.SyncBatchNormalization
        # build the projection head
        with self.scope():
            # if no FCN is passed- build one
            if fcn is None:
                fcn = tf.keras.applications.ResNet50(weights=None, include_top=False)
            self.fcn = fcn
            # Create a Keras model that wraps the base encoder and
            # the projection head
            inpt = tf.keras.layers.Input((fcn.output_shape[-1]), dtype=tf.float32)
            net = tf.keras.layers.Dense(num_hidden, dtype=tf.float32)(inpt)
            if batchnorm:
                net = bnorm(dtype=tf.float32)(net)
            net = tf.keras.layers.Activation("relu", dtype=tf.float32)(net)
            net = tf.keras.layers.Dense(output_dim, use_bias=False, dtype=tf.float32)(net)
            projector = tf.keras.Model(inpt, net)
            # and a model with fcn run through the projector, to use for
            # alignment/uniformity calculation
            inpt = tf.keras.layers.Input((imshape[0], imshape[1], num_channels),
                                         dtype=tf.float32)
            net = fcn(inpt)
            net = tf.keras.layers.GlobalAvgPool2D(dtype=tf.float32)(net)
            net = projector(net)
            full = tf.keras.Model(inpt, net)

        self._models = {"fcn":fcn,
                        "projector":projector,
                        "full":full}

        # build training dataset
        # we need to find the output size of the FCN
        mock_input = np.zeros((1,imshape[0], imshape[1], num_channels), dtype=np.float32)
        outshp = fcn(mock_input).shape # should be rank-4: (1,h,w,d)

        ds = _build_segment_pair_dataset(trainingdata, mean_scale=mean_scale,
                                         num_samples=num_samples,
                                         outputsize=(outshp[1], outshp[2]),
                                         imshape=imshape, batch_size=batch_size,
                                         num_parallel_calls=num_parallel_calls,
                                         norm=norm, num_channels=num_channels,
                                         augment=augment, single_channel=single_channel)
        self._ds = self._distribute_dataset(ds)

        # create optimizer
        self._optimizer = self._build_optimizer(lr, lr_decay, opt_type=opt_type,
                                                decay_type=decay_type,
                                                weight_decay=weight_decay)


        # build training step
        self._training_step = _build_trainstep(fcn, projector,
                                               self._optimizer,
                                               self.strategy,
                                               temperature, weight_decay,
                                               dataparallel)

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
                            mean_scale=mean_scale, num_samples=num_samples,
                            dataparallel=dataparallel, batchnorm=batchnorm,
                            num_hidden=num_hidden, output_dim=output_dim,
                            weight_decay=weight_decay,
                            lr=lr, lr_decay=lr_decay,
                            imshape=imshape, num_channels=num_channels,
                            norm=norm, batch_size=batch_size,
                            num_parallel_calls=num_parallel_calls,
                            single_channel=single_channel, notes=notes,
                            trainer="detcon", strategy=str(strategy),
                            decay_type=decay_type, opt_type=opt_type, **kwargs)

    def _run_training_epoch(self, **kwargs):
        """

        """
        for x1, seg1, x2, seg2 in self._ds:
            lossdict = self._training_step(x1, seg1, x2, seg2)
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
            self._linear_classification_test(avpool=avpool, query_fig=query_fig)



