import numpy as np
import tensorflow as tf
import sklearn.preprocessing, sklearn.cluster, sklearn.decomposition
from sklearn.metrics.cluster import normalized_mutual_info_score
import warnings

from patchwork.feature._models import BNAlexNetFCN
from patchwork.loaders import stratified_training_dataset, dataset
from patchwork.feature._generic import GenericExtractor




def cluster(vecs, pca_dim=256, k=100, init='k-means++', testvecs=None,
            kmeans_max_iter=100, kmeans_batch_size=100):
    """
    Macro to run the entire clustering pipeline
    
    :vec: 2D array of feature vectors
    :pca_dim: number of dimensions to reduce the feature vectors to
                before clustering
    :k: number of clusters for k-means
    :init: how to initialize k-means (check sklearn.cluster.KMeans for details)
    :testvecs: array containing a batch of test outputs
    """
    if pca_dim >= vecs.shape[1]: 
        warnings.warn("hey bro PCA should make the number of dimensions go DOWN. vectors shape %s, pca_dim %s"%(vecs.shape[1], pca_dim))
    
    # create model objects
    scaler = sklearn.preprocessing.StandardScaler()
    pca = sklearn.decomposition.PCA(n_components=pca_dim,
                                     whiten=True)
    kmeans = sklearn.cluster.MiniBatchKMeans(n_clusters=k, init=init, n_init=3,
                                             max_iter=kmeans_max_iter,
                                             batch_size=kmeans_batch_size,
                                             max_no_improvement=100,
                                             compute_labels=True)
                                             #compute_labels=False)
    # fit on the training data
    vecs = scaler.fit_transform(vecs)
    vecs = pca.fit_transform(vecs)
    vecs = sklearn.preprocessing.normalize(vecs, norm="l2")
    kmeans.fit(vecs)
    
    # if there are any empty clusters- randomly reassign the centroid
    # to a jittered randomly-chosen non-empty cluster
    for i in range(k):
        # empty if no labels assigned
        if i not in kmeans.labels_:
            # choose an occupied cluster
            j = np.random.choice(kmeans.labels_)
            new_centroid = kmeans.cluster_centers_[j,:] + np.random.normal(0, 0.1,
                                                  kmeans.cluster_centers_.shape[1])
            kmeans.cluster_centers_[i,:] = new_centroid
        
    
    # if test data was passed- make predictions on that as well
    if testvecs is not None:
        testvecs = scaler.transform(testvecs)
        testvecs = pca.transform(testvecs)
        testvecs = sklearn.preprocessing.normalize(testvecs, norm="l2")
        test_labels = kmeans.predict(testvecs)
    else:
        test_labels = None
    return kmeans.predict(vecs), kmeans.cluster_centers_, test_labels



def _build_model(feature_extractor, imshape=(256,256), num_channels=3,
                 dense=[4096,4096], k=100):
    """
    generate a keras Model for deepcluster training
    
    :feature_extractor: a fully convolutional network to be trained
        for feature extraction
    :imshape: input shape of images
    :num_chanels: number of input channels
    :dense: list of ints; number of neurons in each hidden layer
    :k: int; number of clusters
    
    Returns:
    :prediction_model: fcn through last dense/relu layer, with relu removed
    :training_model: fcn through dense through softmax layer, with softmax removed
    :output_layer: just the final layer (for resetting weights)
    """
    inpt = tf.keras.layers.Input((imshape[0], imshape[1], num_channels))
    feature_tensors = feature_extractor(inpt)
    net = tf.keras.layers.Flatten()(feature_tensors)
    for d in dense:
        net = tf.keras.layers.Dropout(0.5)(net)
        linear = tf.keras.layers.Dense(d)(net)
        net = tf.keras.layers.Activation("relu")(linear)
    output_layer = tf.keras.layers.Dense(k, name="logit_out")
    output = output_layer(net)
    
    prediction_model = tf.keras.Model(inpt, linear)
    training_model = tf.keras.Model(inpt, output)
        
    return prediction_model, training_model, output_layer



def build_deepcluster_training_step():
    @tf.function
    def deepcluster_training_step(x, y, model, opt):
        """
        Basic training function for DeepCluster model.
        """
        print("tracing deepcluster_training_step")
        lossdict = {}
        with tf.GradientTape() as tape:
            y_pred = model(x, training=True)
            loss = tf.reduce_mean(
                    tf.keras.losses.sparse_categorical_crossentropy(y, y_pred,
                                                                    from_logits=True)
                )
            lossdict["training_crossentropy"] = loss
        
        variables = model.trainable_variables
        gradients = tape.gradient(loss, variables)
        opt.apply_gradients(zip(gradients, variables))
    
        return lossdict
    return deepcluster_training_step



class DeepClusterTrainer(GenericExtractor):
    """
    Class for training a DeepCluster model
    """
    modelname = "DeepCluster"

    def __init__(self, logdir, trainingdata, testdata=None, fcn=None, augment=True, 
                 pca_dim=256, k=1000, dense=[4096], mult=1, 
                 kmeans_max_iter=100, kmeans_batch_size=100, lr=0.05, 
                 lr_decay=100000, decay_type="exponential", opt_type="momentum",
                  imshape=(256,256), num_channels=3,
                 norm=255, batch_size=64, num_parallel_calls=None,
                 single_channel=False, notes="",
                 downstream_labels=None):
        """
        :logdir: (string) path to log directory
        :trainingdata: (list) list of paths to training images
        :testdata: (list) filepaths of a batch of images to use for eval
        :fcn: (keras Model) fully-convolutional network to train as feature extractor
        :augment: (dict) dictionary of augmentation parameters, True for defaults or
                False to disable augmentation
        :pca_dim: (int) dimension to reduce FCN outputs to using principal component analysis
        :k: (int) number of clusters
        :dense: (list of ints) number of hidden units in dense layers between the
            FCN and the softmax layer. THERE NEEDS TO BE AT LEAST ONE DENSE LAYER. The
            DeepCluster paper used [4096, 4096].
        :mult: (int) not in paper; multiplication factor to increase
                number of steps/epoch. set to 1 to get paper algorithm
        :kmeans_max_iter: max iterations over dataset for minibatch k-means
        :kmeans_batch_size: batch size for minibatch k-means
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
        self.logdir = logdir
        self.trainingdata = trainingdata
        self._downstream_labels = downstream_labels
        
        self._file_writer = tf.summary.create_file_writer(logdir, flush_millis=10000)
        self._file_writer.set_as_default()
        
        # if no FCN is passed- build one
        if fcn is None:
            fcn = BNAlexNetFCN(num_channels)
        self.fcn = fcn
        self._models = {"fcn":fcn}    
        
        # build model for training    
        prediction_model, training_model, output_layer = _build_model(fcn, 
                                imshape=imshape, num_channels=num_channels, 
                                dense=dense, k=k)
        self._models["full"] = training_model
        self._pred_model = prediction_model
        self._output_layer = output_layer
        
        # create optimizer
        self._optimizer = self._build_optimizer(lr, lr_decay, opt_type=opt_type,
                                                decay_type=decay_type)
        
        # build evaluation dataset
        if testdata is not None:
            self._test_ds, self._test_steps = dataset(testdata,
                                     imshape=imshape,norm=norm,
                                     num_channels=num_channels,
                                     single_channel=single_channel)
            self._test = True
        else:
            self._test = False
        self._test_labels = None
        self._old_test_labels = None
        
        # build prediction dataset for clustering
        ds, num_steps = dataset(trainingdata, imshape=imshape, num_channels=num_channels, 
                 num_parallel_calls=num_parallel_calls, batch_size=batch_size, 
                 augment=False, single_channel=single_channel)
        self._pred_ds = ds
        self._pred_steps = num_steps
        
        # finally, build a Glorot initializer we'll use for resetting the last
        # layer of the network
        self._initializer = tf.initializers.glorot_uniform()
        self._old_cluster_assignments = None
        
        
        # build training step function- this step makes sure that this object
        # has its own @tf.function-decorated training function so if we have
        # multiple deepcluster objects (for example, for hyperparameter tuning)
        # they won't interfere with each other.
        self._training_step = build_deepcluster_training_step()
        self.step = 0
        
        # parse and write out config YAML
        self._parse_configs(augment=augment, k=k, pca_dim=pca_dim, lr=lr, 
                            lr_decay=lr_decay, opt_type=opt_type,
                            mult=mult, dense=dense,
                            kmeans_max_iter=kmeans_max_iter, 
                            kmeans_batch_size=kmeans_batch_size,
                            imshape=imshape, num_channels=num_channels,
                            norm=norm, batch_size=batch_size,
                            num_parallel_calls=num_parallel_calls, 
                            single_channel=single_channel, notes=notes,
                            trainer="deepcluster")
        
        
    def _run_training_epoch(self, **kwargs):
        """
        
        """
        # predict clusters for each data point
        predictions = self._pred_model.predict(self._pred_ds, steps=self._pred_steps)
        
        # if test data was included- also predict outputs for those
        if self._test:
            test_preds = self._pred_model.predict(self._test_ds, steps=self._test_steps)
        else:
            test_preds = None
        
        # run k-means
        y_e, clusters, test_labels = cluster(predictions, 
                                self.config["pca_dim"], 
                                self.config["k"], init='k-means++',
                                kmeans_max_iter=self.config["kmeans_max_iter"],
                                kmeans_batch_size=self.config["kmeans_batch_size"],
                                testvecs=test_preds)
        self._old_test_labels = self._test_labels
        self._test_labels = test_labels
        
        # record the normalized mutual information between these labels and previous
        if self._old_cluster_assignments is not None:
            nmi = normalized_mutual_info_score(y_e, self._old_cluster_assignments,
                                               average_method="arithmetic")
            self._record_scalars(normalized_mutual_information=nmi)
        self._old_cluster_assignments = y_e
        
        
        # reset the weights of the output layer
        new_weights = [self._initializer(x.shape) for x in 
                       self._output_layer.get_weights()]
        self._output_layer.set_weights(new_weights)
        
        # do some training
        train_ds = stratified_training_dataset(self.trainingdata, y_e, 
                                    imshape=self.input_config["imshape"],
                                    num_channels=self.input_config["num_channels"],
                                    num_parallel_calls=self.input_config["num_parallel_calls"],
                                    batch_size=self.input_config["batch_size"], 
                                    mult=self.config["mult"],
                                    augment=self.augment_config,
                                    single_channel=self.input_config["single_channel"])
        
        for x, y in train_ds:
            lossdict = self._training_step(x, y, self._models["full"], 
                                      self._optimizer)
            self._record_scalars(**lossdict)
            self.step += 1
 
    def evaluate(self):
        if self._test:
            if self._test_labels is not None:
                predictions = self._pred_model.predict(self._test_ds,
                                                   steps=self._test_steps)
                test_accuracy = np.mean(predictions == self._test_labels)
                self._record_scalars(epoch_end_test_accuracy=test_accuracy)
                
                if self._old_test_labels is not None:
                    nmi = normalized_mutual_info_score(self._test_labels, self._old_test_labels,
                                                       average_method="arithmetic")
                    self._record_scalars(test_nmi=nmi)
                    
        if self._downstream_labels is not None:
            # choose the hyperparameters to record
            if not hasattr(self, "_hparams_config"):
                from tensorboard.plugins.hparams import api as hp
                hparams = {
                    hp.HParam("pca_dim", hp.IntInterval(0, 1000000)):self.config["pca_dim"],
                    hp.HParam("k", hp.IntInterval(1, 1000000)):self.config["k"],
                    hp.HParam("mult", hp.IntInterval(1, 1000000)):self.config["mult"],
                    }
                for e, d in enumerate(self.config["dense"]):
                    hparams[hp.HParam("dense_%s"%e, hp.IntInterval(1, 1000000))] = d
            else:
                hparams=None
            self._linear_classification_test(hparams)
            
            
            
        
            