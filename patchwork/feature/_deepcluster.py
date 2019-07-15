import numpy as np
import tensorflow as tf
from tqdm import tqdm

#from PIL import Image
import sklearn.preprocessing, sklearn.cluster, sklearn.decomposition

from patchwork.feature._models import BNAlexNetFCN
from patchwork._loaders import stratified_training_dataset, dataset



def cluster(vecs, pca_dim=256, k=100, init='k-means++'):
    """
    Macro to run the entire clustering pipeline
    
    :vec: 2D array of feature vectors
    :pca_dim: number of dimensions to reduce the feature vectors to
                before clustering
    :k: number of clusters for k-means
    :init: how to initialize k-means (check sklearn.cluster.KMeans for details)
    """
    vecs = sklearn.preprocessing.StandardScaler().fit_transform(vecs)
    vecs = sklearn.decomposition.PCA(pca_dim).fit_transform(vecs)
    vecs = sklearn.preprocessing.normalize(vecs, norm="l2")
    
    kmeans = sklearn.cluster.KMeans(n_clusters=k, init=init, n_init=1)
    kmeans.fit(vecs)
    
    return kmeans.predict(vecs), kmeans.cluster_centers_



def _build_model(feature_extractor, imshape=(256,256), num_channels=3,
                 dense=[4096,4096], k=100):
    """
    generate a keras Model for deepcluster training
    
    note- can we set the name for tf.keras.applications models?
    
    :feature_extractor: a fully convolutional network to be trained
        for feature extraction
    :imshape: input shape of images
    :num_chanels: number of input channels
    :dense: list of ints; number of neurons in each hidden layer
    :k: int; number of clusters
    """
    inpt = tf.keras.layers.Input((imshape[0], imshape[1], num_channels))
    feature_tensors = feature_extractor(inpt)
    net = tf.keras.layers.Flatten()(feature_tensors)
    for d in dense:
        net = tf.keras.layers.Dropout(0.5)(net)
        net = tf.keras.layers.Dense(d, activation="relu")(net)
    output_layer = tf.keras.layers.Dense(k, activation="softmax", 
                                         name="softmax_out")
    output = output_layer(net)
    
    model = tf.keras.Model(inpt, net)
    training_model = tf.keras.Model(inpt, output)
    training_model.compile(
    tf.keras.optimizers.RMSprop(1e-3),
    loss=tf.keras.losses.sparse_categorical_crossentropy
    )
        
    return model, training_model, output_layer





def train_deepcluster(filepaths, feature_extractor=None, logdir=None, 
                      epochs=10, pca_dim=256, k=1000, 
                      batch_size=256, mult=10, imshape=(256,256), 
                      num_channels=3, num_parallel_calls=2, dense=[]):
    """
    Train a DeepCluster model.
    
    :filepaths: list of paths to image files
    :feature_extractor: fully-convolutional keras model. If None, build a modified
        AlexNet like what was used in the Deepcluster paper
    :logdir: string; path to directory to write training logs in
    :epochs: number of epochs for training
    :pca_dim: dimension to reduce network outputs to using PCA
    :k: number of clusters for k-means
    :batch_size: batch size
    :mult: not in paper; multiplication factor to increase
        number of steps/epoch. set to 1 to get paper algorithm
    :imshape: shape to resize images to
    :num_channels: number of input channels of images
    :num_parallel_calls: parallelization for input pipeline
    :dense: list of ints for dense layers between feature extractor and
        softmax prediction. set to [4096,4096] to reproduce Caron et al's model
    """
    # no FCN is passed- build one
    if feature_extractor is None:
        feature_extractor = BNAlexNetFCN(num_channels)
    # build model for training    
    model, training_model, output_layer = _build_model(feature_extractor, imshape=imshape, 
                         num_channels=num_channels, dense=[], k=k)
    #[4096,4096]

    if logdir is not None:
        callbacks = [
                tf.keras.callbacks.ModelCheckpoint(logdir+"model.ckpt", monitor="loss"),
                tf.keras.callbacks.TensorBoard(log_dir=logdir, write_graph=False,
                                   update_freq="batch")]
    else:
        callbacks = None
        
    clusters = 'k-means++'
    
    lim = np.sqrt(6/(4096 + k))
    
    for e in tqdm(range(epochs)):
        # every epoch- randomize weights in output layer using Glorot initialization
        lim = np.sqrt(6/(4096 + 100))
        rand_weights = np.random.uniform(-lim, lim, 
                    size=output_layer.weights[0].get_shape().as_list()) 
        biases = np.zeros(output_layer.weights[1].get_shape().as_list())
        output_layer.set_weights([rand_weights, biases])
        # build new model with reset final layer
        #inpt = tf.keras.layers.Input((imshape[0], imshape[1], num_channels))
        #mod_results = model(inpt)
        #output = tf.keras.layers.Dense(k, activation="softmax", name="softmax_out")(mod_results)
        #training_model = tf.keras.Model(inpt, output)
        #training_model.compile(
        #        tf.keras.optimizers.RMSprop(1e-3),
        #        loss=tf.keras.losses.sparse_categorical_crossentropy)

    
        # run predictions
        ds, num_steps = dataset(filepaths, imshape=imshape, num_channels=num_channels, 
                 num_parallel_calls=num_parallel_calls, batch_size=batch_size)
        predictions = model.predict(ds, steps=num_steps)

        # run k-means
        y_e, clusters = cluster(predictions, pca_dim, k, init=clusters)
    
        # do some training
        train_ds, num_steps = stratified_training_dataset(filepaths, y_e, 
                                        imshape=imshape, num_channels=num_channels,
                                        num_parallel_calls=num_parallel_calls,
                                        batch_size=batch_size, mult=mult,
                                        augment=True)
        #output_layer.build()
        training_model.fit(train_ds, steps_per_epoch=num_steps, callbacks=callbacks)
        
        
    return feature_extractor



    