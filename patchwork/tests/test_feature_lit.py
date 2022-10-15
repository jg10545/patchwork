import numpy as np
import tensorflow as tf
import sentencepiece as spm
import io
from patchwork.feature.lit import _build_lit_dataset_from_in_memory_features, save_lit_dataset
from patchwork.feature.lit import load_lit_dataset_from_tfrecords, _wrap_encoder

def _get_encoder(filepath, vocab_size=100):
    model = io.BytesIO()
    spm.SentencePieceTrainer.train(input=filepath, model_writer=model, vocab_size=vocab_size)
    return spm.SentencePieceProcessor(model_proto=model.getvalue())


def test_build_list_dataset_from_in_memory_features(text_sample_path):
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

    encoder = _get_encoder(text_sample_path)

    ds = _build_lit_dataset_from_in_memory_features(taglist, features, encoder, maxlen=maxlen,
                                                    batch_size=batch_size)

    assert isinstance(ds, tf.data.Dataset)
    for x, y in ds:
        break

    assert x.shape == (batch_size, maxlen)
    assert y.shape == (batch_size, d)
    assert x.dtype == tf.int64
    assert y.dtype == tf.float32


def test_build_list_dataset_from_in_memory_features_with_prompt_function(text_sample_path):
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

    encoder = _get_encoder(text_sample_path)

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


def test_save_and_load_tfrecord(test_png_path, tmp_path_factory):
    # generate fake prompt/image pairs
    N = 25
    d = 7
    imfiles = [test_png_path] * N
    prompts = ["foo"] * N
    # fake model for testing
    inpt = tf.keras.layers.Input((None, None, 3))
    net = tf.keras.layers.Conv2D(d, 1)(inpt)
    fcn = tf.keras.Model(inpt, net)
    fcn.count_params()

    # SAVE IT TO TFRECORD FILES
    outdir = str(tmp_path_factory.mktemp("litdata"))
    save_lit_dataset(prompts, imfiles, fcn, outdir, num_shards=2,
                              imshape=(32, 32), num_channels=3,
                              norm=255)
    # LOAD IT BACK
    ds = load_lit_dataset_from_tfrecords(outdir, d, shuffle=5)
    for x,y in ds:
        break

    assert isinstance(ds, tf.data.Dataset)
    assert x.shape == ()
    assert x.dtype == tf.string
    assert y.shape == (d)
    assert y.dtype == tf.float32

def test_save_and_load_tfrecord_with_map_fn(test_png_path, tmp_path_factory, text_sample_path):
    # generate fake prompt/image pairs
    N = 25
    d = 7
    maxlen = 11
    imfiles = [test_png_path] * N
    prompts = ["foo"] * N
    # fake model for testing
    inpt = tf.keras.layers.Input((None, None, 3))
    net = tf.keras.layers.Conv2D(d, 1)(inpt)
    fcn = tf.keras.Model(inpt, net)
    fcn.count_params()

    # SAVE IT TO TFRECORD FILES
    outdir = str(tmp_path_factory.mktemp("litdata"))
    save_lit_dataset(prompts, imfiles, fcn, outdir, num_shards=2,
                              imshape=(32, 32), num_channels=3,
                              norm=255)
    # make an encoder to test loading it back
    encoder = _get_encoder(text_sample_path)
    encode_func = _wrap_encoder(encoder, maxlen)
    # LOAD IT BACK
    ds = load_lit_dataset_from_tfrecords(outdir, d, shuffle=5, map_fn=encode_func)
    for x,y in ds:
        break

    assert isinstance(ds, tf.data.Dataset)
    assert x.shape == (maxlen)
    assert x.dtype == tf.int64
    assert y.shape == (d)
    assert y.dtype == tf.float32


def test_save_and_load_tfrecord_with_map_fn_and_prompt_processor(test_png_path, tmp_path_factory, text_sample_path):
    # generate fake prompt/image pairs
    N = 25
    d = 7
    maxlen = 11
    imfiles = [test_png_path] * N
    prompts = ["foo"] * N
    # fake model for testing
    inpt = tf.keras.layers.Input((None, None, 3))
    net = tf.keras.layers.Conv2D(d, 1)(inpt)
    fcn = tf.keras.Model(inpt, net)
    fcn.count_params()

    # SAVE IT TO TFRECORD FILES
    outdir = str(tmp_path_factory.mktemp("litdata"))
    save_lit_dataset(prompts, imfiles, fcn, outdir, num_shards=2,
                              imshape=(32, 32), num_channels=3,
                              norm=255)
    # make an encoder to test loading it back
    encoder = _get_encoder(text_sample_path)
    def _make_prompt(x):
        return x.upper()
    encode_func = _wrap_encoder(encoder, maxlen, prompt_func=_make_prompt)
    # LOAD IT BACK
    ds = load_lit_dataset_from_tfrecords(outdir, d, shuffle=5, map_fn=encode_func)
    for x,y in ds:
        break

    assert isinstance(ds, tf.data.Dataset)
    assert x.shape == (maxlen)
    assert x.dtype == tf.int64
    assert y.shape == (d)
    assert y.dtype == tf.float32
