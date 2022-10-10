import numpy as np
import tensorflow as tf


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
        # return tf.py_function(_wrap, inp=[x], Tout=tf.string), y
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
