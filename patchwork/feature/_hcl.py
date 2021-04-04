# -*- coding: utf-8 -*-
import tensorflow as tf

from patchwork.feature._generic import GenericExtractor
from patchwork._util import compute_l2_loss, _compute_alignment_and_uniformity

from patchwork.feature._moco import _build_augment_pair_dataset
from patchwork.feature._simclr import _build_embedding_model



def _build_negative_indices(batch_size):
    indices = []
    for i in range(2*batch_size):
        row = []
        for j in range(2*batch_size):
            if (i != j) and (abs(i-j) != batch_size):
                row.append((i,j))
        indices.append(row)
        
    return indices


def _compute_hcl_softmax_probs(pos_exp, neg, temp, beta, tau_plus):
    # DELETE THIS
    gbs = neg.shape[0]
    N = 2*gbs - 2
    # importance sampling weights: shape (gbs, gbs-2)
    imp = tf.exp(beta*neg)
    # partition function: shape (gbs,)
    Z = tf.reduce_mean(imp, -1)
    # reweighted negative logits using importance sampling: shape (gbs,)
    reweight_neg = tf.reduce_sum(imp*tf.exp(neg),-1)/Z
    #
    Ng = (reweight_neg - tau_plus*N*pos_exp)/(1-tau_plus)
    # constrain min value
    Ng = tf.maximum(Ng, N*tf.exp(-1/temp))
    # manually compute softmax
    softmax_prob = pos_exp/(pos_exp+Ng)
    return softmax_prob


def _simclr_softmax_prob(z1, z2, temp):
    """
    Compute SimCLR softmax probability for a pair of batches of normalized
    embeddings. Also compute the batch accuracy.
    
    :z1, z2: normalized embeddings (gbs,d)
    :temp: scalar Gibbs temperature parameter
    
    Returns:
        :softmax_prob:
        :nce_batch_acc:
    """
    # positive logits: just the dot products between corresponding pairs
    # shape (gbs,)
    pos = tf.reduce_sum(z1*z2, -1)/temp
    pos = tf.concat([pos,pos], 0) # from HCL code- line 38 in main.py. shape (2gbs,)
    pos_exp = tf.exp(pos)
            
    z = tf.concat([z1,z2],0)
    # negative samples- first find all pairwise combinations
    # shape (2gbs, 2gbs)
    s = tf.matmul(z, z, transpose_b=True)
    # then remove the comparisons between corresponding pairs
    # shape (2gbs, 2gbs-2)
    gbs = z1.shape[0]
    neg_indices = _build_negative_indices(gbs)
    # not in paper pseudocode or equation, but actual code includes
    # temperature scaling in importance-sampled sum
    neg = tf.gather_nd(s, neg_indices)/temp
    # convert negatives to logits
    # shape (gbs, gbs-2)
    neg_exp = tf.exp(neg)
            
    # COMPUTE BATCH ACCURACY ()
    biggest_neg = tf.reduce_max(neg_exp, -1)
    nce_batch_acc = tf.reduce_mean(tf.cast(pos_exp > biggest_neg, tf.float32))
    # COMPUTE SOFTMAX PROBABILITY (gbs,)
    softmax_prob = pos_exp/(pos_exp + tf.reduce_sum(neg_exp, -1))
    return softmax_prob, nce_batch_acc


def _hcl_softmax_prob(z1, z2, temp, beta, tau_plus):
    """
    Compute HCL softmax probability for a pair of batches of normalized
    embeddings. Also compute the batch accuracy.
    
    :z1, z2: normalized embeddings (gbs,d)
    :temp: scalar Gibbs temperature parameter
    
    Returns:
        :softmax_prob:
        :nce_batch_acc:
    """
    # get the global batch size
    gbs = z1.shape[0]
    N = 2*gbs - 2
    # positive logits: just the dot products between corresponding pairs
    # shape (gbs,)
    pos = tf.reduce_sum(z1*z2, -1)/temp
    pos = tf.concat([pos,pos], 0) # from HCL code- line 38 in main.py. shape (2gbs,)
    pos_exp = tf.exp(pos)
            
    z = tf.concat([z1,z2],0)
    # negative samples- first find all pairwise combinations
    # shape (2gbs, 2gbs)
    s = tf.matmul(z, z, transpose_b=True)
    # then remove the comparisons between corresponding pairs
    # shape (2gbs, 2gbs-2)
    gbs = z1.shape[0]
    neg_indices = _build_negative_indices(gbs)
    # not in paper pseudocode or equation, but actual code includes
    # temperature scaling in importance-sampled sum
    neg = tf.gather_nd(s, neg_indices)/temp
    # convert negatives to logits
    # shape (gbs, gbs-2)
    neg_exp = tf.exp(neg)
            
    # COMPUTE BATCH ACCURACY ()
    biggest_neg = tf.reduce_max(neg_exp, -1)
    nce_batch_acc = tf.reduce_mean(tf.cast(pos_exp > biggest_neg, tf.float32))

    # importance sampling weights: shape (gbs, gbs-2)
    imp = tf.exp(beta*neg)
    # partition function: shape (gbs,)
    Z = tf.reduce_mean(imp, -1)
    # reweighted negative logits using importance sampling: shape (gbs,)
    reweight_neg = tf.reduce_sum(imp*tf.exp(neg),-1)/Z
    #
    Ng = (reweight_neg - tau_plus*N*pos_exp)/(1-tau_plus)
    # constrain min value
    Ng = tf.maximum(Ng, N*tf.exp(-1/temp))
    # manually compute softmax
    softmax_prob = pos_exp/(pos_exp+Ng)
    
    
    return softmax_prob, nce_batch_acc

def _build_trainstep(model, optimizer, strategy, temp=1, tau_plus=0, beta=0, weight_decay=0):
    """
    Build a distributed training step for SimCLR or HCL.
    
    Set tau_plus and beta to 0 for SimCLR parameters.
    
    :model: Keras projection model
    :optimizer: Keras optimizer
    :strategy: tf.distribute.Strategy object
    :temp: temperature parameter
    :tau_plus: HCL class probability parameter
    :beta: HCL concentration parameter
    :weightdecay: L2 loss coefficient. 0 to disable
    
    Returns a distributed training function
    """
    def _step(x,y):
        with tf.GradientTape() as tape:
            loss = 0
            # get replica context- we'll use this to aggregate embeddings
            # across different GPUs
            context = tf.distribute.get_replica_context()
            #print("x,y:", x.shape, y.shape)
            # run images through model and normalize embeddings
            z1 = tf.nn.l2_normalize(model(x, training=True), 1)
            z2 = tf.nn.l2_normalize(model(y, training=True), 1)
            # aggregate projections across replicas. z1 and z2 should
            # now correspond to the global batch size (gbs, d)
            z1 = context.all_gather(z1, 0)
            z2 = context.all_gather(z2, 0)
            #print("z1,z2:", z1.shape, z2.shape)
        
            # SimCLR case
            if (tau_plus == 0)&(beta == 0):
                softmax_prob, nce_batch_acc = _simclr_softmax_prob(z1, z2, temp)
            # HCL case
            elif (tau_plus > 0)&(beta > 0):
                softmax_prob, nce_batch_acc = _hcl_softmax_prob(z1, z2, temp, 
                                                                beta, tau_plus)
            else:
                assert False, "both tau_plus and beta must be nonzero to run HCL"
            """
            # positive logits: just the dot products between corresponding pairs
            # shape (gbs,)
            # CHANGING FOR CONSISTENCY
            #pos = tf.exp(tf.reduce_sum(z1*z2, -1)/temp)
            pos = tf.reduce_sum(z1*z2, -1)/temp
            pos = tf.concat([pos,pos], 0) # from HCL code- line 38 in main.py. shape (2gbs,)
            pos_exp = tf.exp(pos)
            #print("pos:", pos.shape)
            z = tf.concat([z1,z2],0)
            # negative samples- first find all pairwise combinations
            # shape (2gbs, 2gbs)
            s = tf.matmul(z, z, transpose_b=True)
            #print("s:", s.shape)
            # then remove the comparisons between corresponding pairs
            # shape (2gbs, 2gbs-2)
            gbs = z1.shape[0]
            #print("global batch size:", gbs)
            neg_indices = _build_negative_indices(gbs)
            # not in paper pseudocode or equation, but actual code includes
            # temperature scaling in importance-sampled sum
            neg = tf.gather_nd(s, neg_indices)/temp
            # convert negatives to logits
            # shape (gbs, gbs-2)
            neg_exp = tf.exp(neg)
            #print("neg:", neg.shape)
            
            # COMPUTE BATCH ACCURACY
            biggest_neg = tf.reduce_max(neg_exp, -1)
            nce_batch_acc = tf.reduce_mean(tf.cast(pos_exp > biggest_neg, tf.float32))
            
            # SIMCLR CASE
            if (tau_plus == 0)&(beta == 0):
                # manually compute softmax. shape (gbs,)
                softmax_prob = pos_exp/(pos_exp + tf.reduce_sum(neg_exp, -1))
            # HCL CASE (M=1)
            else:
                #N = 2*batch_size - 2
                N = 2*gbs - 2
                
                # importance sampling weights: shape (gbs, gbs-2)
                #imp = tf.exp(beta*neg)
                # partition function: shape (gbs,)
                #Z = tf.reduce_mean(imp, -1)
                # reweighted negative logits using importance sampling: shape (gbs,)
                #reweight_neg = tf.reduce_sum(imp*tf.exp(neg/temp),-1)/Z
                #
                #Ng = (reweight_neg - tau_plus*N*pos)/(1-tau_plus)
                # constrain min value
                #Ng = tf.maximum(Ng, N*tf.exp(-1/temp))
                # manually compute softmax
                #softmax_prob = pos/(pos+Ng)
                
                softmax_prob = _compute_hcl_softmax_probs(pos_exp, neg, temp, beta,
                                                          tau_plus)
            """
            
                
            softmax_loss = tf.reduce_mean(-1*tf.math.log(softmax_prob))
            loss += softmax_loss
            
            if weight_decay > 0:
                l2_loss = compute_l2_loss(model)
                loss += weight_decay*l2_loss
            else:
                l2_loss = 0
            
            
        grad = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grad, model.trainable_variables))
        return {"total_loss":loss, "softmax_loss":softmax_loss, 
                "l2_loss":l2_loss,
               "nce_batch_accuracy":nce_batch_acc}
        
    @tf.function
    def trainstep(x,y):
        per_example_losses = strategy.run(_step, args=(x,y))
        lossdict = {k:strategy.reduce(
                    tf.distribute.ReduceOp.MEAN, 
                    per_example_losses[k], axis=None)
                    for k in per_example_losses}

        return lossdict
    return trainstep


class HCLTrainer(GenericExtractor):
    """
    Class for training an HCL model. SimCLR is a special case of this model,
    for tau_plus = beta = 0.
    
    Based on "Contrastive Learning with Hard Negative Examples" by Robinson et al.
    """
    modelname = "HCL"

    def __init__(self, logdir, trainingdata, testdata=None, fcn=None, 
                 augment=True, temperature=1., beta=0, tau_plus=0,
                 num_hidden=128,
                 output_dim=64, weight_decay=0,
                 lr=0.01, lr_decay=0, decay_type="exponential",
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
        :stratify: pass a list of image labels here to stratify by batch
            during training
        :strategy: if distributing across multiple GPUs, pass a tf.distribute
            Strategy object here
        """
        assert augment is not False, "this method needs an augmentation scheme"
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
                fcn = tf.keras.applications.ResNet50V2(weights=None, include_top=False)
            self.fcn = fcn
            # Create a Keras model that wraps the base encoder and 
            # the projection head
            embed_model = _build_embedding_model(fcn, imshape, num_channels,
                                             num_hidden, output_dim)
        
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
                                                decay_type=decay_type)
        
        
        # build training step
        self._training_step = _build_trainstep(embed_model, self._optimizer, 
                                               self.strategy, temp=temperature,
                                               tau_plus=tau_plus, beta=beta,
                                               weight_decay=weight_decay)
        
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
                            beta=beta, tau_plus=tau_plus,
                            num_hidden=num_hidden, output_dim=output_dim,
                            weight_decay=weight_decay,
                            lr=lr, lr_decay=lr_decay, 
                            imshape=imshape, num_channels=num_channels,
                            norm=norm, batch_size=batch_size,
                            num_parallel_calls=num_parallel_calls,
                            single_channel=single_channel, notes=notes,
                            trainer="hcl", strategy=str(strategy),
                            decay_type=decay_type, opt_type=opt_type)

    def _run_training_epoch(self, **kwargs):
        """
        
        """
        for x, y in self._ds:
            lossdict = self._training_step(x,y)
            self._record_scalars(**lossdict)
            self._record_scalars(learning_rate=self._get_current_learning_rate())
            self.step += 1
             
    def evaluate(self, avpool=True):
        
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

            self._linear_classification_test(hparams,
                        metrics=metrics, avpool=avpool)
        
# -*- coding: utf-8 -*-

