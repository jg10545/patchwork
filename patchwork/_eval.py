import numpy as np
import pandas as pd
import tensorflow as tf
import sklearn.linear_model, sklearn.preprocessing, sklearn.metrics, sklearn.multioutput
from tqdm import tqdm

from patchwork.loaders import dataset
from patchwork._sample import PROTECTED_COLUMN_NAMES


def _add_label_noise(X, rate=0.05):
    """
    Randomly flip values in a matrix of multihot labels
    """
    X = X.copy()
    shift = np.random.choice([0,1], p=[1-rate,rate], size=X.shape)
    return (X+shift)%2


def _split_domains(subset):
    """
    Randomly subset train and test set so that they're from
    different domains
    """
    domains = list(set(subset))
    domains_in_A = len(domains) - 1 #np.random.randint(1, len(domains))
    domainA = set(np.random.choice(domains, replace=False, size=domains_in_A))
    domainB = set(np.array([d for d in domains if d not in domainA]))

    subA = np.array([s in domainA for s in subset])
    subB = np.array([s in domainB for s in subset])

    return subA, subB, domainA, domainB


def _get_accuracy(trainX, trainY, testX, testY, C=1.0):
    """

    """
    estimator = sklearn.linear_model.LogisticRegression(solver='liblinear', max_iter=1000, C=1.0)
    #multiestimator = sklearn.multioutput.MultiOutputClassifier(estimator)
    #multiestimator.fit(trainX, trainY)
    estimator.fit(trainX, trainY)
    train_acc = sklearn.metrics.accuracy_score(trainY, estimator.predict(trainX))
    test_acc = sklearn.metrics.accuracy_score(testY, estimator.predict(testX))
    train_auc = sklearn.metrics.roc_auc_score(trainY, estimator.predict_proba(trainX)[:, 1])
    test_auc = sklearn.metrics.roc_auc_score(testY, estimator.predict_proba(testX)[:, 1])
    return train_acc, test_acc, train_auc, test_auc

def _get_features(fcn, df, imshape=(256 ,256), batch_size=64, num_parallel_calls=None,
                  num_channels=3, norm=255, single_channel=False, augment=False, normalize=False):
    """

    """
    inpt = tf.keras.layers.Input((None, None, num_channels))
    net = fcn(inpt)
    net = tf.keras.layers.GlobalAveragePooling2D()(net)
    model = tf.keras.Model(inpt, net)
    ds, ns = dataset(list(df.filepath.values), imshape=imshape, batch_size=batch_size,
                                num_parallel_calls=num_parallel_calls, num_channels=num_channels,
                                single_channel=single_channel, norm=norm, augment=augment)
    features = model.predict(ds)
    if normalize:
        return sklearn.preprocessing.normalize(features)
    else:
        return features


def _experiment_dict(x_train, y_train, x_test, y_test, C, split_domains, domainA, domainB,
                     k, n, label_noise_frac, rescale, normalize, pos_class_prob):
    train_acc, test_acc, train_auc, test_auc = _get_accuracy(x_train, y_train, x_test, y_test, C=C)
    exptdict = {
        "fcn": k,
        "N": n,
        "label_noise_frac": label_noise_frac,
        "train_accuracy": train_acc,
        "test_accuracy": test_acc,
        "test error": 1 - test_acc,
        "train error": 1 - train_acc,
        "train AUC": train_auc,
        "test AUC": test_auc,
        "rescale":rescale,
        "normalize":normalize,
        "pos_class_prob":pos_class_prob
    }
    if split_domains:
        exptdict["training_domains"] = ", ".join(list(domainA))
        exptdict["test_domains"] = ", ".join(list(domainB))
    return exptdict


def sample_and_evaluate(fcndict, df, category, num_experiments=100, minsize=10, C=1.0, label_noise_frac=0,
                        split_domains=False, image_noise_dict={}, showprogress=True, normalize=False,
                        rescale=False, pos_class_prob=None, usedask=False, **kwargs):
    """
    Train a linear model on top of your feature extractor using random subsets of your
    training set, so that you can visualize how your feature extractor performs as data is added.

    :fcndict: dictionary of keras models containing a fully-convolutional network to compare
    :df: pandas dataframe containing labels; same format you'd use for active learning GUI
    :category:
    :num_experiments: int; number of sampled datasets to train and evaluate models on
    :minsize: minimum size of sampled dataset
    :C: inverse regularization strength for logistic regression (passed to sklearn.linear_model.LogisticRegression())
    :label_noise_frac: randomly select this fraction of training labels and flip their value
    :split_domains: if True, randomly divide the values of the "subset" column into to disjoint sets;
        train on one and evaluate and the other.
    :image_noise_dict: set this to be an augmentation dictionary, and for every experiment features
        will be recomputed using this augmentation as a source of noise.
    :showprogress: whether to use a tqdm progressbar
    :normalize: bool; whether to normalize features to a unit hypersphere
    :rescale: bool; if True, scale so each feature has zero mean and unit variance across the training set
    :kwargs: passed to pw.loaders.dataset()

    Returns a dataframe containing a set of performance measures for each experiment.
    """
    if usedask:
        import dask, dask.diagnostics
    # find all the categories
    #categories = [c for c in df.columns if c not in PROTECTED_COLUMN_NAMES]
    if split_domains:
        subset = df["subset"].values

    if not isinstance(fcndict, dict):
        fcndict = {"fcn": fcndict}

    # boolean array for identifying training and testing points
    notnull = pd.notnull(df[category]).values#.prod(1).astype(bool)
    train_index = (~df.exclude.values) & (~df.validation.values) & notnull
    test_index = (~df.exclude.values) & (~df.validation.values) & notnull

    # get features
    featuredict = {k: _get_features(fcndict[k], df, augment=image_noise_dict,
                                    normalize=normalize, **kwargs)
                   for k in fcndict}

    # run experiments!
    results = []
    if showprogress: progressbar = tqdm(total=num_experiments)
    while len(results) < num_experiments * len(fcndict):
        try:
            # ORDER OF OPERATIONS:
            # 1) rebuild feature vectors if necessary
            # 2) split domains if necessary
            # 3) get train and test features/labels for full dataset
            # 4) random subset of training data
            # 5) corrupt training labels
            # 6) train a model based off each FCN and record results

            # 1) if adding image noise- recompute TRAINING features only
            if len(image_noise_dict) > 0:
                features = {k: _get_features(fcndict[k], df, augment=image_noise_dict,
                                             normalize=normalize, **kwargs)
                            for k in fcndict}
            else:
                features = featuredict

            # 2) split domains if necessary
            if split_domains:
                subA, subB, domainA, domainB = _split_domains(subset)
                expt_train_index = train_index * subA
                expt_test_index = test_index * subB
            else:
                expt_train_index = train_index
                expt_test_index = test_index
                domainA = None
                domainB = None

            # 3) get train and test features and labels
            x_train = {k: features[k][expt_train_index] for k in features}
            x_test = {k: features[k][expt_test_index] for k in features}
            y_train = df[category].values[expt_train_index]
            y_test = df[category].values[expt_test_index]

            # 4) random subset of training data
            N = x_train[list(fcndict.keys())[0]].shape[0]
            n = int(10 ** np.random.uniform(np.log10(minsize), np.log10(N)))
            # sample either uniformly or stratified
            if pos_class_prob is None:
                sample_indices = np.random.choice(np.arange(N), size=n, replace=False)
            else:
                pos_n = max(1, int(n*pos_class_prob))
                neg_n = n - pos_n
                indices = np.arange(N)
                pos_indices = indices[y_train == 1]
                neg_indices = indices[y_train == 0]
                sample_indices = np.concatenate([
                    np.random.choice(pos_indices, size=pos_n, replace=False),
                    np.random.choice(neg_indices, size=neg_n, replace=False)
                ])
            x_train = {k: x_train[k][sample_indices] for k in x_train}
            y_train = y_train[sample_indices]
            # 4a) if rescaling, apply that now
            if rescale:
                for k in x_train:
                    scaler = sklearn.preprocessing.StandardScaler()
                    x_train[k] = scaler.fit_transform(x_train[k])
                    x_test[k] = scaler.transform(x_test[k])

            # 5) add label noise
            y_train = _add_label_noise(y_train, label_noise_frac)

            # check to make our sample has both positive and negative examples
            assert y_train.min() == 0
            assert y_train.max() == 1
            # 6) for each FCN train a model
            for k in fcndict:
                #fcn = fcndict[k]
                if usedask:
                    results.append(dask.delayed(
                        _experiment_dict)(x_train[k], y_train, x_test[k], y_test, C, split_domains, domainA, domainB,
                                         k, n, label_noise_frac, rescale, normalize, pos_class_prob))
                else:
                    results.append(_experiment_dict(x_train[k], y_train, x_test[k], y_test, C, split_domains, domainA, domainB,
                     k, n, label_noise_frac, rescale, normalize, pos_class_prob))
                """
                train_acc, test_acc, train_auc, test_auc = _get_accuracy(x_train[k], y_train, x_test[k], y_test, C=C)
                exptdict = {
                    "fcn": k,
                    "N": n,
                    "label_noise_frac": label_noise_frac,
                    "train_accuracy": train_acc,
                    "test_accuracy": test_acc,
                    "test error": 1 - test_acc,
                    "train error": 1 - train_acc,
                    "train AUC":train_auc,
                    "test AUC":test_auc
                }
                if split_domains:
                    exptdict["training_domains"] = ", ".join(list(domainA))
                    exptdict["test_domains"] = ", ".join(list(domainB))
                results.append(exptdict)"""

            if showprogress: progressbar.update()
        except:
            continue
    if showprogress: progressbar.close()
    if usedask:
        with dask.diagnostics.ProgressBar():
            results = dask.compute(results)[0]
    return pd.DataFrame(results)
