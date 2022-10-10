import numpy as np
import tensorflow as tf
from patchwork.feature.lit import _build_lit_dataset_from_in_memory_features


class FakeEncoder():
    def __init__(self, vocab_size=100):
        self.vocab_size = vocab_size

    def encode(self, x, *args, **kwargs):
        return np.random.randint(0, self.vocab_size, size=len(x))


def test_build_list_dataset_from_in_memory_features():
    # dataset size
    N = 101
    # feature dimension
    d = 13
    # sequence length
    maxlen = 21
    batch_size = 17
    tags = ["foo", "bar", "stuff", "things", "fuzz", "america", "meat"]

    taglist = ["/".join(np.random.choice(tags, size=np.random.randint(1, len(tags)), replace=False))
               for _ in range(N)]

    features = np.random.normal(0, 1, size=(N, d))

    encoder = FakeEncoder()

    ds = _build_lit_dataset_from_in_memory_features(taglist, features, encoder, maxlen=maxlen,
                                                    batch_size=batch_size)

    assert isinstance(ds, tf.data.Dataset)
    for x, y in ds:
        break

    assert x.shape == (batch_size, maxlen)
    assert y.shape == (batch_size, d)
    assert x.dtype == tf.int64
    assert y.dtype == tf.float32


def test_build_list_dataset_from_in_memory_features_with_prompt_function():
    # dataset size
    N = 101
    # feature dimension
    d = 13
    # sequence length
    maxlen = 21
    batch_size = 17
    tags = ["foo", "bar", "stuff", "things", "fuzz", "america", "meat"]

    taglist = ["/".join(np.random.choice(tags, size=np.random.randint(1, len(tags)), replace=False))
               for _ in range(N)]

    features = np.random.normal(0, 1, size=(N, d))

    encoder = FakeEncoder()

    def prompt_function(x):
        return ",".join([y.upper() for y in x.split("/")])

    ds = _build_lit_dataset_from_in_memory_features(taglist, features, encoder, maxlen=maxlen,
                                                    batch_size=batch_size, prompt_func=prompt_function)

    assert isinstance(ds, tf.data.Dataset)
    for x, y in ds:
        break

    assert x.shape == (batch_size, maxlen)
    assert y.shape == (batch_size, d)
    assert x.dtype == tf.int64
    assert y.dtype == tf.float32
