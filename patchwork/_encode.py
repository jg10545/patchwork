import numpy as np
from PIL import Image
import tensorflow as tf



def build_encoder(layers=[32, 64, 128, 256, 512], im_size=(256,256,3)):
    inpt = tf.keras.layers.Input(im_size)
    net = inpt
    for k in layers:
        net = tf.keras.layers.Conv2D(k, 3, strides=2, padding="same",
                                activation=tf.keras.activations.relu)(net)
    return tf.keras.Model(inpt, net, name="encoder")


def build_transition(inpt_size=(8,8,512)):
    inpt =  tf.keras.layers.Input(inpt_size)
    net = tf.keras.layers.Conv2D(256, 1, activation=tf.keras.activations.relu)(inpt)

    shp = net.get_shape().as_list()[1:]

    #net = tf.keras.layers.Flatten()(net)
    net = tf.keras.layers.Dense(256, activation=tf.keras.activations.relu)(net)
    #net = tf.keras.layers.Reshape(shp)(net)
    net = tf.keras.layers.Conv2D(512, 1, activation=tf.keras.activations.relu)(net)

    return tf.keras.Model(inpt, net, name="transition")


def build_decoder(layers=[256, 128, 64, 32, 32], inpt_size=(8,8,512)):
    inpt = tf.keras.layers.Input(inpt_size)
    net = inpt

    for k in layers:
        net = tf.keras.layers.Conv2DTranspose(k, 3, strides=2, padding="same",
                                activation=tf.keras.activations.relu)(net)
    
    net = tf.keras.layers.Conv2D(3, 3, strides=1, padding="same", 
                             activation=tf.keras.activations.sigmoid)(net)
    return tf.keras.Model(inpt, net, name="decoder")




def build_context_encoder():
    N = 256
    mask_start = int(N/4)
    mask = np.zeros((N,N,3), dtype=bool)
    mask[mask_start:3*mask_start, mask_start:3*mask_start,:] = True
    
    def masked_l2_loss(y_true, y_pred):
        m = tf.expand_dims(tf.constant(mask.astype(np.float32)),0)
        m = tf.layers.flatten(m)
    
        return tf.keras.backend.mean(
            m * tf.square(tf.layers.flatten(y_pred) - tf.layers.flatten(y_true)), axis=-1
        )
    
    encoder = build_encoder()
    transition = build_transition()
    decoder = build_decoder()
    
    inpt = tf.keras.layers.Input((256,256,3))
    encoded = encoder(inpt)
    updated = transition(encoded)
    decoded = decoder(updated)

    context_encoder = tf.keras.Model(inpt, decoded)
    context_encoder.compile(tf.keras.optimizers.Adam(1e-3),
                       loss=masked_l2_loss)
    
    return context_encoder, encoder


def train_and_test(filepaths):
    all_files = [x.strip() for x in open(filepaths, "r").readlines()]
    all_ims = np.stack([np.array(Image.open(x).resize((256,256))) for x in all_files])
    test_ims = all_ims[np.arange(all_ims.shape[0]) % 10 == 0,:,:,:]
    train_ims = all_ims[np.arange(all_ims.shape[0]) % 10 != 0,:,:,:]

    train_x = train_ims.copy()
    train_x[:,mask] = 0
    
    train_y = train_ims
    test_y = test_ims

    test_x = test_ims.copy()
    test_x[:,mask] = 0
    
    return train_x, train_y, test_x, test_y