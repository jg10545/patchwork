import numpy as np
import tensorflow as tf
import sklearn.linear_model, sklearn.preprocessing, sklearn.metrics
from tqdm import tqdm

from patchwork.loaders import dataset


def _get_features(fcn, df, imshape=(256 ,256), batch_size=64, num_parallel_calls=None,
                  num_channels=3, norm=255, single_channel=False):
    """

    """
    inpt = tf.keras.layers.Input((None, None, num_channels))
    net = fcn(inpt)
    net = tf.keras.layers.GlobalAveragePooling2D()(net)
    model = tf.keras.Model(inpt, net)
    ds, ns = dataset(list(df.filepath.values), imshape=imshape, batch_size=batch_size,
                                num_parallel_calls=num_parallel_calls, num_channels=num_channels,
                                single_channel=False)
    return model.predict(ds)


def sample_and_evaluate(fcn, df, category, num_experiments=100, minsize=10, **kwargs):
    """

    """
    # get features
    features = _get_features(fcn, df, **kwargs)
    train = features[(~df.exclude.values) & (~df.validation.values) & pd.notnull(df[category]).values]
    test = features[(~df.exclude.values) & (df.validation.values) & pd.notnull(df[category]).values]
    y_train = df2[(~df.exclude.values) & (~df.validation.values) & pd.notnull(df[category]).values][category].values
    y_test = df2[(~df.exclude.values) & (df.validation.values) & pd.notnull(df[category]).values][category].values
    # run experiment
    results = []
    progressbar = tqdm(total=num_experiments)
    while len(results) < num_experiments:
        n = int(10 ** np.random.uniform(np.log10(minsize), np.log10(train.shape[0])))
        indices = np.random.choice(np.arange(train.shape[0]), size=n, replace=False)

        trainingset = train[indices]
        traininglabels = y_train[indices]
        if traininglabels.mean() > 0 and traininglabels.mean() < 1:
            linear = sklearn.linear_model.SGDClassifier(loss="log", max_iter=10000, n_jobs=-1,
                                                        learning_rate="adaptive", eta0=1e-2)
            linear.fit(trainingset, traininglabels)
            probs = linear.predict_proba(test)[:, 1]

            preds = (probs >= 0.5).astype(int)
            metrics = {
                "n": n,
                "num_positive": traininglabels.sum(),
                "num_negative": n - traininglabels.sum(),
                "auc": sklearn.metrics.roc_auc_score(y_test, probs),
                "accuracy": sklearn.metrics.accuracy_score(y_test, preds),
                "f1": sklearn.metrics.f1_score(y_test, preds),
                "precision": sklearn.metrics.precision_score(y_test, preds),
                "recall": sklearn.metrics.recall_score(y_test, preds)
            }
            results.append(metrics)
            progressbar.update()

    progressbar.close()
    return pd.DataFrame(results)
