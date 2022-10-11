import numpy as np
import tensorflow as tf

from patchwork.loaders import _image_file_dataset
from patchwork._tfrecord import save_dataset_to_tfrecords

def _wrap_encoder(encoder, maxlen, prompt_func=None):
    """

    """

    # first wrap all the pieces into a python function
    def _wrap(x):
        x = str(x)
        if prompt_func is not None:
            x = prompt_func(x)
        y = encoder.encode(x, out_type=int, add_bos=True, add_eos=True)
        N = len(y)
        if N > maxlen:
            y = y[:maxlen]
        elif N < maxlen:
            y += [0] * (maxlen - N)
        return np.array(y)

    # then wrap that as a tf.py_func so autograph doesn't get
    # in here and start breaking stuff
    def prompt(x, y):
        return tf.py_function(_wrap, inp=[x], Tout=tf.int64), y

    return prompt


def _build_lit_dataset_from_in_memory_features(prompts, features, encoder, maxlen=72, batch_size=32,
                                               num_parallel_calls=-1, prompt_func=None):
    """
    :prompts: list of string N prompts
    :features: (N,d) numpy array of features
    """
    N = len(prompts)
    d = features.shape[1]

    index = np.arange(N)

    def _gen():
        ordering = np.arange(N)
        np.random.shuffle(ordering)
        for o in ordering:
            yield prompts[o], features[o]

    ds = tf.data.Dataset.from_generator(_gen, output_signature=(
        tf.TensorSpec(shape=(), dtype=tf.string), tf.TensorSpec(shape=(d), dtype=tf.float32)))

    ds = ds.map(_wrap_encoder(encoder, maxlen, prompt_func), num_parallel_calls=num_parallel_calls)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(1)

    return ds


def save_lit_dataset(prompts, filepaths, fcn, outdir, imshape=(256, 256), num_parallel_calls=-1, norm=255,
                     num_channels=3, shuffle=True, single_channel=False,
                     augment=False, batch_size=32, num_shards=10, gzip=True):
    """

    """
    # wrap fcn with a pooling layer so we get a single vector per image
    if single_channel:
        inpt = tf.keras.layers.Input(imshape)
    else:
        inpt = tf.keras.layers.Input((imshape[0], imshape[1], num_channels))
    net = fcn(inpt)
    net = tf.keras.layers.GlobalAveragePooling2D()(net)
    model = tf.keras.Model(inpt, net)
    # load the images- dataset will be (image, prompt)
    ds = _image_file_dataset(filepaths, ys=prompts, imshape=imshape,
                                        num_parallel_calls=num_parallel_calls, norm=norm,
                                        num_channels=num_channels, shuffle=shuffle,
                                        single_channel=single_channel, augment=augment)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(1)

    # run each image through the feature extractor and swap order- dataset will be batches of (prompt, feature)
    def _get_features(x, y):
        return y, model(x)

    ds = ds.map(_get_features, num_parallel_calls=1)
    # unbatch so we get single instances of (prompt, feature)
    ds = ds.unbatch()
    # write to file
    save_dataset_to_tfrecords(ds, outdir, num_shards=num_shards, gzip=gzip)


def load_lit_dataset_from_tfrecords(record_dir, d, shuffle=2048, num_parallel_calls=-1,
                                    map_fn=None, gzip=True):
    """
    Load a directory structure of tfrecord files (like you'd build with save_to_tfrecords)
    into a tensorflow dataset for training a LiT model.

    Assumes tfrecord files are saved with .tfrecord or .snapshot.

    :record_dir: top-level directory containing record files
    :d: embedding dimension for image features
    :shuffle: if not False, size of shuffle queue
    :num_parallel_calls: number of parallel readers/mappers for loading and parsing
        the dataset
    :map_fn: function to map across dataset during loading
    :gzip: whether tfrecord was saved using GZIP compression
    """
    if gzip:
        comp = "GZIP"
    else:
        comp = "NONE"

    element_spec = (
        tf.TensorSpec(shape=(), dtype=tf.string),
        tf.TensorSpec(shape=(d), dtype=tf.float32)
    )
    # note that this function may change in the future
    ds = tf.data.experimental.load(record_dir, element_spec, compression=comp)
    # if a map function was included, map across the dataset
    if map_fn:
        ds = ds.map(map_fn, num_parallel_calls=num_parallel_calls)
    if shuffle:
        ds = ds.shuffle(shuffle)
    return ds
