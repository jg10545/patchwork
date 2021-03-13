# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
#import tensorflow.keras.backend as K

from patchwork.feature._generic import GenericExtractor
from patchwork._util import compute_l2_loss, _compute_alignment_and_uniformity
from patchwork.feature._moco import _build_augment_pair_dataset, _build_logits
from patchwork.feature._moco import copy_model, exponential_model_update

from tqdm import tqdm


def _compute_buffer_gradient(buffer, q, all_logits, adv_tau, weight_decay):
    """
    Implementing equation 6 from the paper
    
    :all_logits: [batch_size, K+1]
    """
    batch_size = all_logits.shape[0]
    # equation 5
    # I initially misread this equation the paper- the distribution is
    # over query vectors, not negative vectors, so the softmax should be computed
    # along axis 0. [batch_size, K+1]
    probs = tf.nn.softmax(all_logits/adv_tau, 0)
    #probs = tf.nn.softmax(all_logits[:,1:]/adv_tau, 0)
    # just the negative probabilities [batch_size, K]
    #negprobs = probs[:,1:]
    negprobs = probs[:,batch_size:]
    # equation 6
    # negprobs^T x q = [K, batch_size] x [batch_size, d] = [K,d]
    adv_gradient = tf.matmul(negprobs, q, transpose_a=True)/(batch_size*adv_tau)
    # include weight decay
    update = -1*adv_gradient + weight_decay*buffer
    return update


def _build_adco_training_step(model, mo_model, buffer, opt, 
                              adv_opt, tau=0.12,  adv_tau=0.07, weight_decay=0):
    """
    Function to build tf.function for a MoCo training step. Basically just follow
    Algorithm 1 in He et al's paper.
    """
    
    @tf.function
    def training_step(img1, img2):
        print("tracing training step")
        batch_size = img1.shape[0]
        outdict = {}
        buff = tf.nn.l2_normalize(buffer, axis=1)
        #buff = buffer
        # create labels ("correct" class is 0)- (N,)
        #labels = tf.zeros((batch_size,), dtype=tf.int32)
        print("batch comparison labels")
        labels = tf.range(batch_size, dtype=tf.int64)
        # START SHUFFLE STUFF
        print("EVERY DAY I'M SHUFFLING")
        # compute averaged embeddings. tensor is (N,d)
        # my shot at an alternative to shuffling BN
        #a = int(batch_size/4)
        b = int(batch_size/2)
        #c = int(3*batch_size/4)
        
        k1 = mo_model(img2[:b,:,:,:], training=True)
        k2 = mo_model(img2[b:,:,:,:], training=True)
        #k = tf.nn.l2_normalize(tf.concat([k1,k2], 0), 1)
        print("using batchnorm trick from momentum^2")
        foo = tf.nn.l2_normalize(tf.concat([k1,k2], 0), 1)
        k = tf.nn.l2_normalize(mo_model(img2, training=False), 1)
        #k1 = mo_model(tf.concat([img2[:a,:,:,:], img2[c:,:,:,:]], 0),
        #              training=True)
        #k2 = mo_model(img2[a:c,:,:,:], training=True)
        #k = tf.nn.l2_normalize(tf.concat([
        #            k1[:a,:], k2, k1[a:,:]], axis=0
        #        ), axis=1)
        with tf.GradientTape() as tape:
            #q1 = model(img1[:b,:,:,:], training=True)
            #q2 = model(img1[b:,:,:,:], training=True)
            #q = tf.nn.l2_normalize(tf.concat([q1,q2], 0), axis=1)
            # skip shuffle buffer with query network so we don't get any
            # funny business passing gradients back through the batchnorms
            q = model(img1, training=True)
            q = tf.nn.l2_normalize(q, 1)
            # END SHUFFLE STUFF
            
            # compute logits
            print("including negative comparisons from within the batch")
            all_logits = _build_logits(q, k, buff, compare_batch=True)
            # compute crossentropy loss
            loss = tf.reduce_mean(
                    tf.nn.sparse_softmax_cross_entropy_with_logits(
                            labels, all_logits/tau))
                        
            if weight_decay > 0:
                l2_loss = compute_l2_loss(model)
                outdict["l2_loss"] = l2_loss
                loss += weight_decay*l2_loss
        # ------------- UPDATE MODEL ----------------
        variables = model.trainable_variables
        gradients = tape.gradient(loss, variables)
        opt.apply_gradients(zip(gradients, variables))
        # also compute the "accuracy"; what fraction of the batch has
        # the key as the largest logit. from figure 2b of the MoCHi paper
        #nce_batch_accuracy = tf.reduce_mean(tf.cast(tf.argmax(all_logits, 
        #                                                      axis=1)==0, tf.float32))
        nce_batch_accuracy = tf.reduce_mean(tf.cast(tf.argmax(all_logits, 
                                                              axis=1)==labels, tf.float32))
        
        # ------------- UPDATE BUFFER ----------------
        #buffergrad = _compute_buffer_gradient(buffer, q, all_logits, adv_tau, 
        #                                      weight_decay)
        print("using autograd for buffer update")
        with tf.GradientTape() as advtape:
            buff = tf.nn.l2_normalize(buffer, axis=1)
            all_logits = _build_logits(q, k, buff, compare_batch=True)
            adv_loss = -1*tf.reduce_mean(
                    tf.nn.sparse_softmax_cross_entropy_with_logits(
                            labels, all_logits/adv_tau))
            adv_loss += weight_decay*tf.reduce_mean(buffer**2)
        #adv_opt.apply_gradients(zip([buffergrad], [buffer]))
        grad = advtape.gradient(adv_loss, [buffer])
        adv_opt.apply_gradients(zip(grad, [buffer]))
        #print("extra normalization step on buffer")
        #buffer.assign(tf.nn.l2_normalize(buffer, axis=1))
        
        outdict["loss"] = loss
        outdict["nce_batch_accuracy"] = nce_batch_accuracy
        outdict["foo"] = tf.reduce_mean(foo)
        #outdict["rms_buffergrad"] = tf.math.sqrt(tf.reduce_mean(buffergrad**2))
        return outdict
    return training_step



class AdversarialContrastTrainer(GenericExtractor):
    """
    Class for training an Adversarial Contrast model.
    
    Based on "AdCo: Adversarial Contrast for Efficient Learning of
    Unsupervised Representations from Self-Trained Negative
    Adversaries" by Hu et al
    """
    modelname = "MomentumContrast"

    def __init__(self, logdir, trainingdata, testdata=None, fcn=None, 
                 augment=True, negative_examples=65536, alpha=0.999,
                 tau=0.12, adv_tau=0.02, output_dim=128, num_hidden=2048, 
                 weight_decay=1e-4,
                 lr=0.03, adv_lr=3.0, lr_decay=0, decay_type="exponential",
                 opt_type="momentum", adaptive=False,
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
        :negative_examples: number of adversarial negative examples
        :tau:.temperature parameter for noise-contrastive loss
        :adv_tau: temperature parameter for updating negative examples
        :batches_in_buffer:
        :num_hidden: number of neurons in the projection head's hidden layer (from the MoCoV2 paper)
        :weight_decay: L2 loss weight; 0 to disable
        :lr: (float) initial learning rate
        :adv_lr: (float) initial learning rate for adversarial updates
        :lr_decay: (int) number of steps for one decay period (0 to disable)
        :decay_type: (string) how to decay the learning rate- "exponential" (smooth exponential decay), "staircase" (non-smooth exponential decay), or "cosine"
        :opt_type: (str) which optimizer to use; "momentum" or "adam"
        :adaptive:
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
        :strategy:
        """
        assert augment is not False, "this method needs an augmentation scheme"
        self.logdir = logdir
        self.trainingdata = trainingdata
        self._downstream_labels = downstream_labels
        self.strategy = strategy
        self._adaptive = adaptive
        
        self._file_writer = tf.summary.create_file_writer(logdir, flush_millis=10000)
        self._file_writer.set_as_default()
        
        # if no FCN is passed- build one
        with self.scope():
            #print("low variance initializer")
            #initializer = tf.random_normal_initializer(0., 0.02)
            if fcn is None:
                fcn = tf.keras.applications.ResNet50V2(weights=None, include_top=False)
            self.fcn = fcn
            # from "technical details" in paper- after FCN they did global pooling
            # and then a dense layer. i assume linear outputs on it.
            inpt = tf.keras.layers.Input((None, None, num_channels))
            features = fcn(inpt)
            pooled = tf.keras.layers.GlobalAvgPool2D()(features)
            # MoCoV2 paper adds a hidden layer
            dense = tf.keras.layers.Dense(num_hidden)(pooled)
            #dense = tf.keras.layers.Dense(num_hidden, kernel_initializer=initializer)(pooled)
            #print("ADDING BATCHNORM TO PROJECTION HEAD")
            #dense = tf.keras.layers.BatchNormalization()(dense)
            dense = tf.keras.layers.Activation("relu")(dense)
            # end
            #outpt = tf.keras.layers.Dense(output_dim, kernel_initializer=initializer)(dense)
            outpt = tf.keras.layers.Dense(output_dim)(dense)
            #print("ADDING ONE MORE GODDAMN BATCHNORM")
            #outpt = tf.keras.layers.BatchNormalization()(outpt)
            full_model = tf.keras.Model(inpt, outpt)
            mo_model = copy_model(full_model)
        
            self._models = {"fcn":fcn, "full":full_model,
                            "momentum_encoder":mo_model}
        
        # build training dataset
        ds = _build_augment_pair_dataset(trainingdata, 
                            imshape=imshape, batch_size=batch_size,
                            num_parallel_calls=num_parallel_calls, 
                            norm=norm, num_channels=num_channels, 
                            augment=augment, single_channel=single_channel)
        self._ds = self._distribute_dataset(ds)
        
        # create optimizers for both steps
        self._optimizer = self._build_optimizer(lr, lr_decay, opt_type=opt_type,
                                                decay_type=decay_type)
        #print("using vanilla SGD for negatives")
        self._adv_optimizer = self._build_optimizer(adv_lr, lr_decay,
                                                    opt_type="momentum",
                                                    decay_type=decay_type)
        
        # build buffer
        with self.scope():
            self._buffer = tf.Variable(np.zeros((negative_examples, output_dim), 
                                            dtype=np.float32),
                                       aggregation=tf.VariableAggregation.MEAN)
        
        # build update step for momentum contrast update
        @tf.function
        def momentum_update_step():
            return exponential_model_update(mo_model, full_model, alpha)
        self._momentum_update_step = momentum_update_step
        # build  and distribute both training steps
        step_fn = _build_adco_training_step(full_model, mo_model, self._buffer,
                                            self._optimizer,
                                            self._adv_optimizer,
                                            tau=tau,
                                            adv_tau=adv_tau,
                                            weight_decay=weight_decay)
        
        self._training_step = self._distribute_training_function(step_fn)
        #self._adv_step = self._distribute_training_function(adv_step_fn)
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

        self.step = 0
        
        
        # parse and write out config YAML
        self._parse_configs(augment=augment, 
                            negative_examples=negative_examples,
                            tau=tau, adv_tau=adv_tau, 
                            output_dim=output_dim, num_hidden=num_hidden,
                            weight_decay=weight_decay,
                            lr=lr, adv_lr=adv_lr,  lr_decay=lr_decay,
                            opt_type=opt_type,
                            imshape=imshape, num_channels=num_channels,
                            norm=norm, batch_size=batch_size,
                            num_parallel_calls=num_parallel_calls, 
                            single_channel=single_channel, notes=notes,
                            trainer="adco")
        self._prepopulate_buffer()
        
    def _prepopulate_buffer(self):
        i = 0
        bs = self.input_config["batch_size"]
        K = self.config["negative_examples"]
        #print("normalizing buffer during prepopulate step")
        while i*bs < K:
            for x,y in self._ds:
                k = self._models["full"](y, training=True)
                _ = self._buffer[bs*i:bs*(i+1),:].assign(tf.nn.l2_normalize(k,1))
                #_ = self._buffer[bs*i:bs*(i+1),:].assign(k,1)
                i += 1
                if i*bs >= K:
                    break
        
    def _run_training_epoch(self, **kwargs):
        """
        
        """
        #if self._adaptive:
        #    self._run_adaptive_training_epoch(**kwargs)
            
        #else:
        if True:
            #buffer_hist = np.zeros(self._buffer.shape[0])
            #buffer_hist_counter = 0
            for x, y in self._ds:
                #if self.step % 2 == 0:
                if True:
                    self._record_scalars(moco_sq_diff=self._momentum_update_step())
                    losses = self._training_step(x,y)
                    self._record_scalars(**losses)
                    #self._record_scalars(learning_rate=self._get_current_learning_rate())
                    #_ = self._adv_step(x,y) # REMEMBER TO DELETE THIS IF SWAPPING BACK
                #else:
                    #losses = self._adv_step(x,y)
                    #self._record_scalars(mean_adv_gradient_norms=tf.reduce_mean(tf.math.sqrt(tf.reduce_sum(losses["grads"]**2, 1))))
                    #buffer_hist += np.sqrt(tf.reduce_sum(losses["grads"]**2, 1).numpy()/self._buffer.shape[1])
                    #buffer_hist_counter += 1
                    
                self.step += 1
            #self._record_hists(buffer_log_gradients=np.log10(buffer_hist/buffer_hist_counter+1e-10))
                
                
    def pretrain(self, epochs=1, rebuild_buffer_every=1000):
        """
        EXPERIMENTAL
        """
        for e in tqdm(range(epochs)):
            # run only main training step
            for x,y in self._ds:
                losses = self._training_step(x,y)
                self._record_scalars(**losses)
                self._record_scalars(learning_rate=self._get_current_learning_rate())
                self.step += 1
                
                if self.step % rebuild_buffer_every == 0:
                    self._prepopulate_buffer()
            self.evaluate()
                
    def _run_adaptive_training_epoch(self, **kwargs):
        assert False, "deprecated"
        if not hasattr(self, "_adaptive_rolling_mean"):
            self._adaptive_rolling_mean = 0
            
        buffer_hist = np.zeros(self._buffer.shape[0])
        buffer_hist_counter = 0
        for x,y in self._ds:
            # if the accuracy is too low, update the feature extractor
            if self._adaptive_rolling_mean < self._adaptive:
                losses = self._training_step(x,y)
                self._record_scalars(**losses)
                self._record_scalars(update_buffer=0)
            # if the accuracy is too high, update the buffer of negatives
            else:
                losses = self._adv_step(x,y)
                self._record_scalars(update_buffer=1,
                                     nce_batch_accuracy=losses["nce_batch_accuracy"])
                buffer_hist += np.sqrt(tf.reduce_sum(losses["grads"]**2, 1).numpy()/self._buffer.shape[1])
                buffer_hist_counter += 1
            nce_acc = losses["nce_batch_accuracy"].numpy()
            self._adaptive_rolling_mean = 0.9*self._adaptive_rolling_mean +\
                                            0.1*nce_acc
            self._record_scalars(adaptive_rolling_mean=self._adaptive_rolling_mean)
            self._record_hists(buffer_log_gradients=np.log10(buffer_hist/buffer_hist_counter+1e-10))
            self.step += 1
                
            
            
 
    def evaluate(self):
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
                    hp.HParam("tau", hp.RealInterval(0., 10000.)):self.config["tau"],
                    hp.HParam("adv_tau", hp.RealInterval(0., 10000.)):self.config["adv_tau"],
                    hp.HParam("negative_examples", hp.IntInterval(1, 1000000)):self.config["negative_examples"],
                    hp.HParam("output_dim", hp.IntInterval(1, 1000000)):self.config["output_dim"],
                    hp.HParam("num_hidden", hp.IntInterval(1, 1000000)):self.config["num_hidden"],
                    hp.HParam("weight_decay", hp.RealInterval(0., 10000.)):self.config["weight_decay"],
                    hp.HParam("lr", hp.RealInterval(0., 10000.)):self.config["lr"],
                    hp.HParam("adv_lr", hp.RealInterval(0., 10000.)):self.config["adv_lr"]
                    }
                for k in self.augment_config:
                    if isinstance(self.augment_config[k], float):
                        hparams[hp.HParam(k, hp.RealInterval(0., 10000.))] = self.augment_config[k]
            else:
                hparams=None
            self._linear_classification_test(hparams, metrics=metrics)
        
        
    def load_weights(self, logdir):
        """
        Update model weights from a previously trained model
        """
        super().load_weights(logdir)
        self._prepopulate_buffer()
            
        
