import numpy as np
import tensorflow as tf

from PIL import Image
import sklearn.preprocessing, sklearn.cluster, sklearn.decomposition





def img_loader(f, imshape=(256,256)):
    """
    Load and prepare image from a filename
    """
    img_raw = tf.read_file(f)
    decoded = tf.image.decode_image(img_raw)
    dim_removed = tf.squeeze(decoded)
    resized = tf.image.resize_image_with_crop_or_pad(dim_removed, 
                                                imshape[0], imshape[1])
    recast = tf.cast(resized, tf.float32)
    return recast/255


def augment(im):
    im = tf.image.random_brightness(im, 0.1)
    im = tf.image.random_contrast(im, 0.5, 1.2)
    im = tf.image.random_flip_left_right(im)
    im = tf.image.random_flip_up_down(im)
    # some augmentation can put values outside unit interval
    im = tf.minimum(im, 1)
    im = tf.maximum(im, 0)
    return im


def predict_dataset(fp, batch_size=256):
    """
    return a tf dataset that iterates over all the images once
    
    :fp: list of strings containing paths to image files
    :batch_size: just what you think it is
    
    Returns
    :ds: tf.data.Dataset object to iterate over data
    :num_steps: number of steps (for passing to tf.keras.Model.fit())
    """
    ds = tf.data.Dataset.from_tensor_slices(np.array(fp))
    ds = ds.map(img_loader, num_parallel_calls=2)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(1)
    
    num_steps = int(np.ceil(len(fp)/batch_size))
    return ds, num_steps


def training_dataset(fp, y, batch_size=256, mult=10):
    """
    Build a dataset that provides stratified samples over
    the last set of k-means clusters
    
    :fp: list of strings containing paths to image files
    :y: array of cluster assignments- should have same length as fp
    :batch_size: just what you think it is
    :mult: not in paper; multiplication factor to increase
        number of steps/epoch. set to 1 to get paper algorithm
        
    Returns
    :ds: tf.data.Dataset object to iterate over data
    :num_steps: number of steps (for passing to tf.keras.Model.fit())
    """
    # sample indices to use
    indices = np.arange(len(fp))
    K = y.max()+1
    samples_per_cluster = mult*int(len(fp)/K)
    
    sampled_indices = []
    sampled_labels = []
    # for each cluster
    for k in range(K):
        # find indices of samples assigned to it
        cluster_inds = indices[y == k]
        # only sample if at least one is assigned. note that
        # the deepcluster paper takes an extra step here.
        if len(cluster_inds) > 0:
            samps = np.random.choice(cluster_inds, size=samples_per_cluster,
                            replace=True)
            sampled_indices.append(samps)
            sampled_labels.append(k*np.ones(len(samps), dtype=np.int64))
    # concatenate sampled indices for each cluster
    sampled_indices = np.concatenate(sampled_indices, 0)    
    sampled_labels = np.concatenate(sampled_labels, 0)
    # and shuffle their order together
    reorder = np.random.choice(np.arange(len(sampled_indices)),
                          size=len(sampled_indices), replace=False)
    sampled_indices = sampled_indices[reorder]
    sampled_labels = sampled_labels[reorder]
    
    # NOW CREATE THE DATASET
    ds = tf.data.Dataset.from_tensor_slices((np.array(fp)[sampled_indices], 
                                             sampled_labels))
    ds = ds.map(lambda x,y: (img_loader(x), y), num_parallel_calls=2)
    ds = ds.map(lambda x,y: (augment(x), y), num_parallel_calls=2)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(1)
    
    num_steps = int(np.ceil(len(sampled_indices)/batch_size))
    return ds, num_steps



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
    softmax_out = tf.keras.layers.Softmax(name="softmax_out")(feature_vectors)
    model = tf.keras.Model(inpt, [feature_vectors, softmax_out])
    
    return model





def train_model(model, filepaths, logdir, num_epochs=10, pca_dim=256, k=100, 
                batch_size=256, mult=10):
    """
    
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
        ds, num_steps = predict_dataset(filepaths, batch_size)
        predictions = model.predict(ds, steps=num_steps)
    
        # run k-means
        y_e, clusters = cluster(predictions[0], pca_dim, k, init=clusters)
    
        # do some training
        train_ds, num_steps = training_dataset(filepaths, y_e, batch_size=batch_size)
        model.fit(train_ds, steps_per_epoch=num_steps, callbacks=callbacks)
        
        
    return model