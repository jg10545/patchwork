import numpy as np
from PIL import Image
import tensorflow as tf

from patchwork._layers import ChannelWiseDense
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def build_encoder(layers=[32, 64, 128, 256, 512], im_size=(256,256,3)):
    inpt = tf.keras.layers.Input(im_size)
    net = inpt
    for k in layers:
        net = tf.keras.layers.Conv2D(k, 3, strides=2, padding="same")(net)
        net = tf.keras.layers.LeakyReLU(alpha=0.2)(net)
        net = tf.keras.layers.BatchNormalization()(net)
    return tf.keras.Model(inpt, net, name="encoder")


def build_decoder(layers=[256, 128, 64, 32, 32], inpt_size=(8,8,512)):
    inpt = tf.keras.layers.Input(inpt_size)
    net = inpt

    for k in layers:
        net = tf.keras.layers.Conv2DTranspose(k, 3, strides=2, padding="same",
                                activation=tf.keras.activations.relu)(net)
        net = tf.keras.layers.BatchNormalization()(net)
    
    net = tf.keras.layers.Conv2D(3, 3, strides=1, padding="same", 
                             activation=tf.keras.activations.sigmoid)(net)
    return tf.keras.Model(inpt, net, name="decoder")


def build_discriminator(layers=[32, 64, 128, 256, 512], im_size=(256,256,3)):
    inpt = tf.keras.layers.Input(im_size)
    net = inpt
    for k in layers:
        net = tf.keras.layers.Conv2D(k, 3, strides=2, padding="same",
                                activation=tf.keras.activations.relu)(net)
        #net = tf.keras.layers.Conv2D(k, 3, strides=2, padding="same")(net)
        #net = tf.keras.layers.LeakyReLU()(net)
        net = tf.keras.layers.BatchNormalization()(net)
    net = tf.keras.layers.GlobalMaxPool2D()(net)
    net = tf.keras.layers.Dense(1, activation=tf.keras.activations.sigmoid,
                               name="disc_pred")(net)
    return tf.keras.Model(inpt, net, name="discriminator")




def build_context_encoder():
    """
    Build a context encoder, mostly following Pathak et al. FOR NOW assumes images
    are (256,256,3).
    
    Returns the context encoder as well as an encoder model.
    """
    # initialize encoder and decoder objects
    encoder = build_encoder()
    decoder = build_decoder()
    # inputs for this model: the image and a mask (which we'll use to only
    # count loss from the masked area)
    inpt = tf.keras.layers.Input((256,256,3), name="inpt")
    inpt_mask = tf.keras.layers.Input((256,256,3), name="inpt_mask")
    # Pathak's structure runs images through the encoder, then a dense
    # channel-wise layer, then a 1x1 Convolution before decoding.
    encoded = encoder(inpt)
    updated = ChannelWiseDense()(encoded)
    dropout = tf.keras.layers.Dropout(0.5)(updated)
    conv1d = tf.keras.layers.Conv2D(512,1)(dropout)
    decoded = decoder(conv1d)
    # create a masked output to compare with ground truth (which should
    # already have it's unmasked areas set to 0)
    masked_decoded = tf.keras.layers.Multiply(name="masked_decoded")(
                                    [inpt_mask, decoded])
    
    # NOW FOR THE ADVERSARIAL PART
    discriminator = build_discriminator()
    discriminator.compile(tf.keras.optimizers.Adam(1e-4), 
                          loss=tf.keras.losses.binary_crossentropy)
    disc_pred = discriminator(decoded)
    

    context_encoder = tf.keras.Model([inpt, inpt_mask], 
                                     [decoded, masked_decoded, disc_pred])
    context_encoder.compile(tf.keras.optimizers.Adam(1e-3),
                            loss={"masked_decoded":tf.keras.losses.mse,
                                 "discriminator":tf.keras.losses.binary_crossentropy},
                            #loss_weights={"masked_decoded":0.99, "discriminator":0.01})
                           loss_weights={"masked_decoded":0.999, "discriminator":0.001})

    
    return context_encoder, encoder, discriminator


def train_and_test(filepaths):
    """
    Load a set of images into memory from file, split 90-10 into train/test
    sets, and build a generator that will do augmentation on the training set.
    
    :filepaths: list of strings pointing to image files (FOR NOW assuming all
        are 256x256x3
    
    Returns
    :train_generator: function to return a generator for keras.Model.fit_generator()
    :val_data: nested tuples of the form ((x, mask),y) to pass to the validation_data
        kwarg of keras.Model.fit_generator()
    
    """
    N = 256
    mask_start = int(N/4)
    mask = np.zeros((N,N,3), dtype=bool)
    mask[mask_start:3*mask_start, mask_start:3*mask_start,:] = True
    
    all_files = [x.strip() for x in open(filepaths, "r").readlines()]
    all_ims = np.stack([np.array(Image.open(x).resize((256,256))) 
                        for x in all_files])/255
    test_ims = all_ims[np.arange(all_ims.shape[0]) % 10 == 0,:,:,:]
    train_ims = all_ims[np.arange(all_ims.shape[0]) % 10 != 0,:,:,:]

    train = train_ims
    test_y = test_ims.copy() 
    test_y[:,~mask] = 0

    test_x = test_ims.copy()
    test_x[:,mask] = 0
    
    def train_generator(batch_size=64):
        datagen = ImageDataGenerator(rotation_range=10, 
                            width_shift_range=0.05,
                            height_shift_range=0.05,
                            zoom_range=0.2,
                            horizontal_flip=True,
                            vertical_flip=True,
                            data_format="channels_last",
                            brightness_range=[0.8,1.2])

        _gen = datagen.flow(train, batch_size=batch_size)
        while True:
            aug_imgs = next(_gen)/255
            masks = np.stack([mask]*aug_imgs.shape[0])
            x = aug_imgs.copy()
            x[masks] = 0
            y = aug_imgs.copy()
            y[~masks] = 0
            yield ((x, masks), y)
    
    #return train_x, train_y, test_x, test_y, mask # added mask
    return train_generator, ((test_x, np.stack([mask]*test_x.shape[0])), test_y)#, mask # added mask