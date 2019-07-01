import numpy as np
import tensorflow as tf

from PIL import Image
import sklearn.preprocessing, sklearn.cluster, sklearn.decomposition

from patchwork._loaders import stratified_training_dataset, predict_dataset



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



def build_model(feature_extractor, input_channels=3):
    """
    generate a keras Model for deepcluster training
    
    note- can we set the name for tf.keras.applications models?
    
    :feature_extractor: a fully convolutional network to be trained
        for feature extraction
    :input_channels: number of input channels
    """
    inpt = tf.keras.layers.Input((None, None, input_channels))
    feature_tensors = feature_extractor(inpt)
    feature_vectors = tf.keras.layers.GlobalMaxPool2D(name="feat_vec")(feature_tensors)
    #dropout = tf.keras.layers.Dropout(0.5)(feature_vectors)
    softmax_out = tf.keras.layers.Softmax(name="softmax_out")(feature_vectors)
    model = tf.keras.Model(inpt, [feature_vectors, softmax_out])
    
    return model





def deepcluster_train(model, filepaths, logdir, num_epochs=10, pca_dim=256, k=100, 
                batch_size=256, mult=10, imshape=(256,256), num_channels=3,
                num_parallel_calls=2):
    """
    Train a DeepCluster model.
    
    :model: an untrained fully-convolutional Keras model
    :filepaths: list of paths to image files
    :logdir: string; path to directory to write training logs in
    :num_epochs: number of epochs for training
    :pca_dim: dimension to reduce network outputs to using PCA
    :k: number of clusters for k-means
    :batch_size: batch size
    :mult: not in paper; multiplication factor to increase
        number of steps/epoch. set to 1 to get paper algorithm
    """
    model.compile(
    tf.keras.optimizers.RMSprop(1e-3),
    loss={"softmax_out":tf.keras.losses.sparse_categorical_crossentropy}
    )
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(logdir+"model.ckpt", monitor="loss"),
        tf.keras.callbacks.TensorBoard(log_dir=logdir, write_graph=False,
                                   update_freq="batch")]
    clusters = 'k-means++'
    
    for e in range(num_epochs):
        print("STARTING EPOCH %s"%e)
    
        # run predictions
        ds, num_steps = predict_dataset(filepaths, imshape, num_channels, 
                 num_parallel_calls, batch_size)
        predictions = model.predict(ds, steps=num_steps)
    
        # run k-means
        y_e, clusters = cluster(predictions[0], pca_dim, k, init=clusters)
    
        # do some training
        train_ds, num_steps = stratified_training_dataset(filepaths, y_e, 
                                        imshape=imshape, num_channels=num_channels,
                                        num_parallel_calls=num_parallel_calls,
                                        batch_size=batch_size, mult=mult,
                                        augment=True)
        #train_ds, num_steps = training_dataset(filepaths, y_e, batch_size=batch_size)
        model.fit(train_ds, steps_per_epoch=num_steps, callbacks=callbacks)
        
        
    return model



    