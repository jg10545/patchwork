import numpy as np
import tensorflow as tf
from patchwork._fine_tuning_models import GlobalPooling, ConvNet, MultiscalePooling



def test_GlobalPooling_with_maxpool():
    pool = GlobalPooling()
    model, _ = pool.build((None, None, 2), None)

    testarr = np.zeros((1,5,7,2))
    testarr[0,:,:,1] = 1

    test_out = model(testarr).numpy()

    assert isinstance(model, tf.keras.Model)
    assert len(model.layers) == 2
    assert (test_out == np.array([[0.,1.]])).all()



def test_GlobalPooling_with_avpool():
    pool = GlobalPooling()
    pool.pooling_type = "average pool"
    model, _ = pool.build((None, None, 2), None)

    testarr = np.zeros((1,5,4,2))
    testarr[0,:,:2,1] = 1

    test_out = model(testarr).numpy()

    assert isinstance(model, tf.keras.Model)
    assert len(model.layers) == 2
    assert (test_out == np.array([[0.,0.5]])).all()


def test_ConvNet():
    c = ConvNet()

    c.layers = "10,p,11"
    model, _ = c.build((None, None, 7), None)

    assert isinstance(model, tf.keras.Model)
    assert len(model.layers) == 7
    assert model.output_shape[1] == 11


def test_ConvNet_with_dropout():
    c = ConvNet()

    c.dropout_rate = 0.5
    c.layers = "7,p,d,11,p,d,13"
    model, _ = c.build((None, None, 7), None)

    assert isinstance(model, tf.keras.Model)
    assert len(model.layers) == 12
    assert model.output_shape[1] == 13

def test_ConvNet_with_separable_convolutions():
    c = ConvNet()

    c.layers = "5,13"
    c.separable_convolutions = True
    model, _ = c.build((None, None, 7), None)

    assert isinstance(model, tf.keras.Model)
    assert len(model.layers) == 6
    assert model.output_shape[1] == 13


def test_multiscalepooling_with_layer_index_list():
    inpt = tf.keras.layers.Input((None, None, 3))
    net = tf.keras.layers.Conv2D(3, 1, name="foo")(inpt)
    net = tf.keras.layers.Conv2D(5, 1, name="bar")(net)
    net = tf.keras.layers.Conv2D(7, 1, name="foobar")(net)
    testfcn = tf.keras.Model(inpt, net)

    mp = MultiscalePooling()
    mp.layers = "1,2"
    finetuning, newfcn = mp.build((None, None, 7), testfcn)
    assert finetuning.output_shape[-1] == 3 + 5


def test_multiscalepooling_with_layer_name_regex():
    inpt = tf.keras.layers.Input((None, None, 3))
    net = tf.keras.layers.Conv2D(3, 1, name="foo")(inpt)
    net = tf.keras.layers.Conv2D(5, 1, name="bar")(net)
    net = tf.keras.layers.Conv2D(7, 1, name="foobar")(net)
    testfcn = tf.keras.Model(inpt, net)

    mp = MultiscalePooling()
    mp.layers = "foo"
    finetuning, newfcn = mp.build((None, None, 7), testfcn)
    assert finetuning.output_shape[-1] == 3 + 7
