# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf

from patchwork.feature._generic import GenericExtractor, _TENSORBOARD_DESCRIPTIONS
from patchwork._augment import augment_function
from patchwork.loaders import load_dataset_from_tfrecords
from patchwork._util import compute_l2_loss, _compute_alignment_and_uniformity
from patchwork.feature._contrastive import _build_augment_pair_dataset, _contrastive_loss
from patchwork.feature._simclr import _build_embedding_model, _gather
from patchwork.loaders import _generate_imtypes, _build_load_function


_DESCRIPTIONS = {
    "nt_xent_loss":"Contrastive crossentropy loss",
    "nce_batch_acc":"training accuracy for the contrastive learning task",
    "weight_diff":"total squared difference in the values of weights between the encoder and momentum encoder"
}
for d in _TENSORBOARD_DESCRIPTIONS:
    _DESCRIPTIONS[d] = _TENSORBOARD_DESCRIPTIONS[d]



def copy_model(mod):
    """
    Clone a Keras model and set the new model's trainable weights to the old
    model's weights
    """
    new_model = tf.keras.models.clone_model(mod)

    for orig, clone in zip(mod.trainable_variables, new_model.trainable_variables):
        clone.assign(orig)
    return new_model



def _update_queue(z,Q):
    """
    Update the support queue to include a new batch of embeddings
    :z: (N,d) tensor
    :Q: (queue_size, d) tf.Variable
    """
    queue_size = Q.shape[0]
    Q.assign(tf.concat([z,Q],0)[:queue_size,:])


def exponential_model_update(slow, fast, alpha=0.999, update_bn=False):
    """
    Update the weights of a "slow" network as a single-exponential
    average of a "fast" network. Return the sum of squared
    differences.

    :slow: Keras model containing exponentially-smoothed weights
    :fast: Keras model with same structure as slow, but updated more
        quickly from a different mechanism
    :alpha: exponential smoothing parameter
    """
    rolling_sum = 0
    for s, f in zip(slow.trainable_variables, fast.trainable_variables):
        rolling_sum += tf.reduce_sum(tf.square(s-f))
        s.assign(alpha*s + (1-alpha)*f)

    if update_bn:
        for s,f in zip(slow.variables, fast.variables):
            # looking for names like 'conv5_block1_0_bn_1/moving_mean:0' and
            # 'conv5_block3_3_bn_1/moving_variance:0'
            if ("bn" in s.name)&("moving" in s.name):
                s.assign(alpha*s + (1-alpha)*f)
    return rolling_sum




def _build_logits(q, k, buffer, N=0, s=0, s_prime=0, margin=0, compare_batch=False):
    """
    Compute logits for momentum contrast, optionally including MoCHi
    hard negatives

    :q: [batch_size, embed_dim] embeddings from online model
    :k: [batch_size, embed_dim] embeddings from momentum encoder
    :buffer: [embed_dim, buffer_size] negative examples
    :N: mochi N param
    :s: mochi s param
    :s_prime: mochi s_prime param
    :margin: margin parameter from EqCo paper
    :compare_batch: if True, include pairwise comparisons between q and k
        across batch

    Returns
    :logits: [batch_size, buffer_size+1] if compare_batch is False
        [batch_size, batch_size+buffer_size] if compare_batch is True
    """
    """
    if compare_batch:
        assert margin==0, "NOT IMPLEMENTED"
        positive_logits = tf.matmul(q, k, transpose_b=True)
    else:
        # compute positive logits- (batch_size,1)
        positive_logits = tf.squeeze(
                tf.matmul(tf.expand_dims(q,1),
                      tf.expand_dims(k,1), transpose_b=True),
                axis=-1) - margin"""
    positive_logits = tf.matmul(q,k, transpose_b=True)
    # and negative logits- (batch_size, buffer_size)
    negative_logits = tf.matmul(q, buffer, transpose_b=True)
    # assemble positive and negative- (batch_size, buffer_size+1)
    all_logits = tf.concat([positive_logits, negative_logits], axis=1)
    # from MoChi paper
    if (N > 0)&(s > 0):
        # find the top-N negative logits- the N hardest "naturally
        # occurring" negatives. inds is (batch_size, N)
        vals, inds = tf.math.top_k(negative_logits, k=N)
        # test for case where we're sampling more than N combinations
        while inds.shape[1] < max(s, s_prime):
            inds = tf.concat([inds, inds], axis=1)
        # sample twice from the list of hard negative indices. each is (batch_size,s)
        inds1 = tf.transpose(tf.random.shuffle(tf.transpose(inds)))[:,:s]
        inds2 = tf.transpose(tf.random.shuffle(tf.transpose(inds)))[:,:s]
        # gather the actual embeddings corresponding to the sampled indices.
        # each is (batch_size, s, embed_dim)
        gathered1 = tf.gather(buffer, inds1, axis=0)
        gathered2 = tf.gather(buffer, inds2, axis=0)
        # and a mixing coefficient for each s
        alpha = tf.random.uniform((1,s,1), minval=1, maxval=1)
        # combine the sampled embeddings using the mixing coefficients, and normalize.
        # will still be (batch_size, s, embed_dim)
        mixed_negatives = tf.stop_gradient(
                            tf.nn.l2_normalize(alpha*gathered1+(1-alpha)*gathered2, 2))
        # hard negatives by mixing query vectors
        if s_prime > 0:
            # sample from list of hard negative indices- (batch_size, s_prime)
            inds3 = tf.transpose(tf.random.shuffle(tf.transpose(inds)))[:,:s_prime]
            # gather the actual embeddings- (batch_size, s_prime, embed_dim)
            gathered3 = tf.gather(buffer, inds3, axis=0)
            # and a mixing coefficient for each one
            beta = tf.random.uniform((1,s_prime,1), minval=0, maxval=0.5)
            # and tile query vectors to combine with negatives
            q_tiled = tf.tile(tf.expand_dims(q,1), tf.constant([1,s_prime,1]))
            hardest_negatives = tf.stop_gradient(
                            tf.nn.l2_normalize(beta*q_tiled + (1-beta)*gathered3, 2))

        # now, with gradients, compute the logits between query embeddings and our
        # synthetic negatives- shape (batch_size, s)
        hnm_logits = tf.squeeze(tf.matmul(mixed_negatives, tf.expand_dims(q, -1)), -1)
        # and concatenate on the end of our block'o'logits: (batch_size, buffer_size+s+s_prime+1)
        if s_prime > 0:
            hardest_logits = tf.squeeze(tf.matmul(hardest_negatives, tf.expand_dims(q, -1)), -1)
            all_logits = tf.concat([all_logits, hnm_logits, hardest_logits], axis=1)
        else:
            all_logits = tf.concat([all_logits, hnm_logits], axis=1)
    return all_logits


def _build_momentum_contrast_training_step(model, mo_model, optimizer, buffer, alpha=0.999, tau=0.07, weight_decay=0,
                                           N=0, s=0, s_prime=0, margin=0):
    """
    Function to build tf.function for a MoCo training step. Basically just follow
    Algorithm 1 in He et al's paper.
    """

    @tf.function
    def training_step(img1, img2):
        print("tracing training step")
        batch_size = img1.shape[0]
        # compute averaged embeddings. tensor is (N,d)
        # my shot at an alternative to shuffling BN
        a = int(batch_size/4)
        b = int(batch_size/2)
        c = int(3*batch_size/4)

        k1 = mo_model(tf.concat([img2[:a,:,:,:], img2[c:,:,:,:]], 0),
                      training=True)
        k2 = mo_model(img2[a:c,:,:,:], training=True)
        k = tf.nn.l2_normalize(tf.concat([
                k1[:a,:], k2, k1[a:,:]], axis=0
            ), axis=1)
        with tf.GradientTape() as tape:
            # compute normalized embeddings for each
            # separately-augmented batch of pairs of images. tensor is (N,d)
            q1 = model(img1[:b,:,:,:], training=True)
            q2 = model(img1[b:,:,:,:], training=True)
            q = tf.nn.l2_normalize(tf.concat([q1,q2], 0), axis=1)
            # compute MoCo and/or MoCHi logits
            all_logits = _build_logits(q, k, buffer, N, s, s_prime, margin)
            # create labels (correct class is 0)- (N,)
            #labels = tf.zeros((batch_size,), dtype=tf.int32)
            # create labels (correct class is the batch index)
            labels = tf.range((batch_size), dtype=tf.int32)
            # compute crossentropy loss
            xent_loss = tf.reduce_mean(
                    tf.nn.sparse_softmax_cross_entropy_with_logits(
                            labels, all_logits/tau))
            if weight_decay > 0:
                l2_loss = compute_l2_loss(model)
            else:
                l2_loss = 0
            loss = xent_loss + weight_decay*l2_loss

        # update fast model
        variables = model.trainable_variables
        gradients = tape.gradient(loss, variables)
        optimizer.apply_gradients(zip(gradients, variables))
        # update slow model
        weight_diff = exponential_model_update(mo_model, model, alpha)

        # update buffer
        #i = step % batches_in_buffer
        #_ = buffer[batch_size*i:batch_size*(i+1),:].assign(k)
        _update_queue(k, buffer)

        # also compute the "accuracy"; what fraction of the batch has
        # the key as the largest logit. from figure 2b of the MoCHi paper
        #nce_batch_accuracy = tf.reduce_mean(tf.cast(tf.argmax(all_logits,
        #                                                      axis=1)==0, tf.float32))
        nce_batch_accuracy = tf.reduce_mean(tf.cast(tf.argmax(all_logits,
                                                              axis=1)==tf.cast(labels,tf.int64), tf.float32))

        return {"loss":loss, "weight_diff":weight_diff,
                "l2_loss":l2_loss, "nt_xent_loss":xent_loss,
                "nce_batch_acc":nce_batch_accuracy}
    return training_step





class MomentumContrastTrainer(GenericExtractor):
    """
    Class for training a Momentum Contrast model.

    Based on "Momentum Contrast for Unsupervised Visual Representation
    Learning" by He et al.
    """
    modelname = "MomentumContrast"

    def __init__(self, logdir, trainingdata, testdata=None, fcn=None,
                 augment=True, K=4096, alpha=0.999,
                 temperature=0.1, output_dim=128, num_hidden=2048,
                 batchnorm=True, num_projection_layers=2,
                 copy_weights=False, weight_decay=0,
                 N=0, s=0, s_prime=0, margin=0,
                 lr=0.01, lr_decay=100000, decay_type="exponential",
                 opt_type="momentum",
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
        :K: memory buffer size
        :alpha: momentum parameter for updating the momentum encoder
        :temperature: temperature parameter for noise-contrastive loss
        :output_dim: dimension for features to be projected into for NCE loss
        :num_hidden: number of neurons in the projection head's hidden layer (from the MoCoV2 paper)
        :batchnorm: bool, whether to use batch normalization in projection head
        :num_projection_layers: int; number of layers in projection head
        :copy_weights: if True, copy the query model weights at the beginning as well
            as the structure.
        :weight_decay: L2 loss weight; 0 to disable
        :N: MoCHi N parameter; sample from top-N logits. 0 to disable.
        :s: MoCHi s parameter; create this many synthetic hard negatives
        :s_prime: MoCHi s_prime parameter; this many query-mixed hard negatives
        :margin: EqCo margin parameter; shift positive logits by this amount (0 to disable)
        :lr: (float) initial learning rate
        :lr_decay: (int) number of steps for one decay period (0 to disable)
        :decay_type: (string) how to decay the learning rate- "exponential" (smooth exponential decay), "staircase" (non-smooth exponential decay), or "cosine"
        :opt_type: (str) which optimizer to use; "momentum" or "adam"
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
                fcn = tf.keras.applications.ResNet50V2(weights=None, include_top=False)
            self.fcn = fcn
            full_model = _build_embedding_model(fcn, imshape, num_channels, num_hidden, output_dim,
                           batchnorm=batchnorm, num_projection_layers=num_projection_layers)
            # from "technical details" in paper- after FCN they did global pooling
            # and then a dense layer. i assume linear outputs on it.
            #inpt = tf.keras.layers.Input((None, None, num_channels))
            #features = fcn(inpt)
            #pooled = tf.keras.layers.GlobalAvgPool2D()(features)
            # MoCoV2 paper adds a hidden layer
            #dense = tf.keras.layers.Dense(num_hidden)(pooled)
            #dense = tf.keras.layers.BatchNormalization()(dense)
            #dense = tf.keras.layers.Activation("relu")(dense)
            #outpt = tf.keras.layers.Dense(output_dim)(dense)
            #full_model = tf.keras.Model(inpt, outpt)

            if copy_weights:
                momentum_encoder = copy_model(full_model)
            else:
                momentum_encoder = tf.keras.models.clone_model(full_model)
            self._models = {"fcn":fcn,
                        "full":full_model,
                        "momentum_encoder":momentum_encoder}

        # build training dataset
        ds = _build_augment_pair_dataset(trainingdata,
                            imshape=imshape, batch_size=batch_size,
                            num_parallel_calls=num_parallel_calls,
                            norm=norm, num_channels=num_channels,
                            augment=augment, single_channel=single_channel)
        self._ds = self._distribute_dataset(ds)

        # build buffer
        with self.scope():
            d = output_dim
            self._buffer = tf.Variable(np.zeros((K, d), dtype=np.float32))
        self._prepopulate_buffer(batch_size, K)


        # create optimizer
        self._optimizer = self._build_optimizer(lr, lr_decay, opt_type=opt_type,
                                                decay_type=decay_type)



        # build training step
        trainstep = _build_momentum_contrast_training_step(
                full_model,
                momentum_encoder,
                self._optimizer,
                self._buffer,
                alpha, temperature, weight_decay,
                N, s, s_prime, margin)
        self._training_step = self._distribute_training_function(trainstep)
        # build evaluation dataset
        if testdata is not None:
            self._test_ds = _build_augment_pair_dataset(testdata,
                            imshape=imshape, batch_size=batch_size,
                            num_parallel_calls=num_parallel_calls,
                            norm=norm, num_channels=num_channels,
                            augment=augment, single_channel=single_channel)
            self._test = True
        else:
            self._test = False
        self._test_labels = None
        self._old_test_labels = None


        self.step = 0


        # parse and write out config YAML
        self._parse_configs(augment=augment, K=K,
                            alpha=alpha, temperature=temperature, output_dim=output_dim,
                            num_hidden=num_hidden, weight_decay=weight_decay,
                            batchnorm=batchnorm, num_projection_layers=num_projection_layers,
                            N=N, s=s, s_prime=s_prime, margin=margin,
                            lr=lr, lr_decay=lr_decay, opt_type=opt_type,
                            imshape=imshape, num_channels=num_channels,
                            norm=norm, batch_size=batch_size,
                            num_parallel_calls=num_parallel_calls,
                            single_channel=single_channel, notes=notes,
                            trainer="moco", **kwargs)

    def _prepopulate_buffer(self, batch_size, K):
        # define a function that takes a batch, runs it through
        # the momentum encoder, aggregates across GPUs, and
        # adds to the memory buffer
        def _buffer_step(x,y):
            k = tf.nn.l2_normalize(
                    self._models["momentum_encoder"](y, training=True),
                axis=1)
            k = _gather(k)
            _update_queue(k,self._buffer)
            return {}
        # distribute the buffer-filling function
        buffer_step = self._distribute_training_function(_buffer_step)
        # now pull from the training set until the buffer should be full
        i = 0
        for x,y in self._ds:
            buffer_step(x,y)
            i += 1
            if i >= K/batch_size:
                break

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


    #def load_weights(self, logdir):
    #    """
    #    Update model weights from a previously trained model
    #    """
    #    super().load_weights(logdir)
    #    self._prepopulate_buffer()


