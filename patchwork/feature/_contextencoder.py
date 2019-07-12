"""

            Code for Pathak et al's Context Encoder

"""
import numpy as np
import tensorflow as tf
import tensorflow.contrib

from patchwork._augment import _augment
from patchwork._loaders import _image_file_dataset
from patchwork._util import _load_img
from patchwork._layers import ChannelWiseDense
from patchwork.feature._ce_models import build_encoder, build_decoder, build_discriminator


def mask_generator(H,W,C):
    """
    Generates random rectangular masks
    
    :H,W,C: height, width, and number of channels
    """
    dh = int(H/2)
    dw = int(W/2)
    while True:
        mask = np.zeros((H,W,C), dtype=bool)
        xmin = np.random.randint(0, dw)
        ymin = np.random.randint(0, dh)
        xmax = xmin + dw
        ymax = ymin + dh
        mask[ymin:ymax, xmin:xmax,:] = True        
        yield mask

def _make_test_mask(H,W,C):
    """
    Generate a mask for a (H,W,C) image that crops out the center fourth.
    """
    mask = np.zeros((H,W,C), dtype=bool)
    mask[int(0.25*H):int(0.75*H), int(0.25*W):int(0.75*W),:] = True
    return mask

def maskinator(img, mask):
    """
    Input an image and a mask; output the image, mask
    masked image (mask removed) and target image 
    (everything but the mask removed)
    """
    mask_float = tf.cast(mask, tf.float32)
    antimask = 1 - mask_float
    
    masked_img = img * antimask
    target_img = img * mask_float
    return img, mask, masked_img, target_img


def _build_context_encoder_dataset(filepaths, input_shape=(256,256,3), norm=255,
                                   shuffle_queue=1000, num_parallel_calls=4,
                                   batch_size=32, prefetch=True):
    """
    Build a tf.data.Dataset object to use for training.
    """
    # first build a Dataset that generates masks
    def _gen():
        return mask_generator(*input_shape)
    mask_ds = tf.data.Dataset.from_generator(_gen,
                                        output_types=(tf.bool),
                                        output_shapes=input_shape)
    # now a Dataset to load images
    img_ds = _image_file_dataset(filepaths, imshape=input_shape[:2], 
                                 channels=input_shape[2], norm=norm,
                                 num_parallel_calls=num_parallel_calls)
    img_ds = img_ds.shuffle(shuffle_queue)
    img_ds = img_ds.map(_augment, num_parallel_calls=num_parallel_calls)
    # combine the image and mask datasets
    zipped_ds = tf.data.Dataset.zip((img_ds, mask_ds))
    # precompute masked images for context encoder input
    masked_img_ds = zipped_ds.map(maskinator, num_parallel_calls=num_parallel_calls)
    masked_batched_ds = masked_img_ds.batch(batch_size)
    if prefetch:
        masked_batched_ds = masked_batched_ds.prefetch(1)
    return masked_batched_ds
    
    
def _build_test_dataset(filepaths, input_shape=(256,256,3), norm=255):
    """
    Load a set of images into memory from file and mask the centers to
    use as a test set.
    
    :filepaths: list of strings pointing to image files 
    :input_shape: dimensions of input images
    :norm: normalizing value for images
    
    Returns
    img_arr, mask, masked_img, target_img
    """
    img_arr = np.stack([_load_img(f, norm=norm, channels=input_shape[2], 
                                  resize=input_shape[:2]) for f in filepaths])
    mask = _make_test_mask(*input_shape)
    mask = np.stack([mask for _ in range(img_arr.shape[0])])
    
    masked_imgs = img_arr.copy()
    masked_imgs[mask] = 0
    target_imgs = img_arr.copy()
    target_imgs[~mask] = 0
    return img_arr, mask, masked_imgs, target_imgs

    

def build_inpainting_network(input_shape=(256,256,3), disc_loss=0.001, 
                             learn_rate=1e-4, encoder=None, 
                             decoder=None, discriminator=None):
    """
    Build an inpainting network as described in the supplementary 
    material of Pathak et al's paper.
    
    :input_shape: 3-tuple giving the shape of images to be inpainted
    :disc_loss: weight for the discriminator component of the loss function.
        1-disc_loss will be applied to the reconstruction loss
    :learn_rate: learning rate for inpainter. discriminator will be set to
        1/10th of this
    :encoder: encoder model (if not specified one will be built)
    :decoder: decoder model
    :discriminator: discriminator model
    
    Returns inpainter, encoder, and discriminator models.
    """
    # initialize encoder and decoder objects
    if encoder is None: 
        encoder = build_encoder(input_shape[-1])
    if decoder is None:
        decoder = build_decoder(num_channels=input_shape[-1])

    inpt = tf.keras.layers.Input(input_shape, name="inpt")
    inpt_mask = tf.keras.layers.Input(input_shape, name="inpt_mask")
    # Pathak's structure runs images through the encoder, then a dense
    # channel-wise layer, then dropout and a 1x1 Convolution before decoding.
    encoded = encoder(inpt)
    dense = ChannelWiseDense()(encoded)
    dropout = tf.keras.layers.Dropout(0.5)(dense)
    conv1d = tf.keras.layers.Conv2D(512,1)(dropout)
    decoded = decoder(conv1d)
    # create a masked output to compare with ground truth (which should
    # already have it's unmasked areas set to 0)
    masked_decoded = tf.keras.layers.Multiply(name="masked_decoded")(
                                    [inpt_mask, decoded])
    
    # NOW FOR THE ADVERSARIAL PART
    if discriminator is None:
        discriminator = build_discriminator(input_shape[-1])
    discriminator.compile(tf.keras.optimizers.Adam(0.1*learn_rate), 
                          loss=tf.keras.losses.binary_crossentropy)
    discriminator.trainable = False
    disc_pred = discriminator(decoded)
    

    inpainter = tf.keras.Model([inpt, inpt_mask], 
                               [decoded, masked_decoded, disc_pred])
    inpainter.compile(tf.keras.optimizers.Adam(learn_rate),
                      loss={"masked_decoded":tf.keras.losses.mse,
                            "discriminator":tf.keras.losses.binary_crossentropy},
                            loss_weights={"masked_decoded":1-disc_loss, 
                                          "discriminator":disc_loss})
    return inpainter, encoder, discriminator




def train_context_encoder(trainfiles, testfiles=None, inpainter=None, 
                          discriminator=None, num_epochs=1000, 
                          num_test_images=10, logdir=None, 
                          input_shape=(256,256,3), norm=255,
                          num_parallel_calls=4, batch_size=32):
    """
    Train your very own context encoder
    
    :trainfiles: list of paths to training files
    :testfiles: list of paths to test files
    :inpainter, discriminator: if not specified, will be auto-generated
    """
    if inpainter is None or discriminator is None:
        inpainter, enc, discriminator = build_inpainting_network(
                input_shape=input_shape, disc_loss=0.001)
        
    assert tf.executing_eagerly(), "eager execution needs to be enabled"
    # build training generator
    train_ds = _build_context_encoder_dataset(trainfiles, input_shape=input_shape, 
                                norm=norm, shuffle_queue=1000, 
                                num_parallel_calls=num_parallel_calls,
                                batch_size=batch_size, prefetch=True)
    
    if testfiles is not None:
        assert logdir is not None, "need a place to store test results"
        test = True
        test_ims, test_mask, test_masked_ims, test_target_ims = \
                                    _build_test_dataset(testfiles, 
                                            input_shape=input_shape, norm=norm)
        summary_writer = tf.contrib.summary.create_file_writer(logdir, 
                                                       flush_millis=10000)
        summary_writer.set_as_default()
        global_step = tf.train.get_or_create_global_step()

    else:
        test = False

    # combined training loop
    for e in range(num_epochs):
        # for each step in the epoch:
        for img, mask, masked_img, target_img in train_ds:
            # prepare batch inputs
            masked_img = masked_img.numpy()
            img = img.numpy()
            mask = mask.numpy()
            target_img = target_img.numpy()
            # effective batch size
            bs = img.shape[0]
            ce_labels = np.ones(bs)
            disc_labels = np.concatenate([
                    np.zeros(bs),
                    np.ones(bs)
            ])
            # run training step on inpainting network
            inpainter.train_on_batch((masked_img, mask), 
                                 (target_img, ce_labels))
            # generate reconstructed images
            reconstructed_images = inpainter.predict((masked_img, mask))
            # make discriminator batch
            disc_batch_x = np.concatenate([reconstructed_images[0], img], 0)
            # run discriminator training step
            discriminator.train_on_batch(disc_batch_x, disc_labels)
    
        # at the end of the epoch, evaluate on test data. this has two steps:
        #  -compute loss metrics on all of the test set
        #  -visualize reconstructions for the first few test examples
        if test:
            # evaluation- list of 3 values: ['loss', 'decoder_loss', 'discriminator_loss']
            test_results = inpainter.evaluate((test_masked_ims, test_mask), 
                               [test_target_ims, np.ones(test_ims.shape[0])])
            # predict on the first few
            preds = inpainter.predict((test_masked_ims[:num_test_images],
                                       test_mask[:num_test_images]))[0]
            # for the visualization in tensorboard: replace the unmasked areas
            # with the input image as a guide to the eye
            preds = preds*test_mask[:num_test_images] + \
                        test_ims[:num_test_images]*(1-test_mask[:num_test_images])
            predviz = np.concatenate([test_masked_ims[:num_test_images], preds], 
                             2).astype(np.float32)
            
            with tf.contrib.summary.always_record_summaries():
                tf.contrib.summary.scalar("total_loss", test_results[0])
                tf.contrib.summary.scalar("decoder_loss", test_results[1])
                tf.contrib.summary.scalar("discriminator_loss", test_results[2])
                for j in range(num_test_images):
                    tf.contrib.summary.image("img_%i"%j, 
                                     np.expand_dims(predviz[j,:,:,:],0))
        global_step.assign_add(1)
    return inpainter, discriminator