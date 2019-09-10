"""

            Code for Pathak et al's Context Encoder

"""
import numpy as np
import tensorflow as tf
import tensorflow.contrib
import tensorflow.keras.backend as K

from patchwork._augment import _augment
from patchwork._loaders import _image_file_dataset
from patchwork._util import _load_img
from patchwork._layers import ChannelWiseDense
from patchwork.feature._models import build_encoder, build_decoder, build_discriminator


def mask_generator(H,W,C):
    """
    Generates random rectangular masks
    
    :H,W,C: height, width, and number of channels
    """
    dh = int(H/2)
    dw = int(W/2)
    while True:
        mask = np.zeros((H,W,C), dtype=np.float32)
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
    #mask = np.zeros((H,W,C), dtype=bool)
    mask = np.zeros((H,W,C), dtype=np.float32)
    #mask[int(0.25*H):int(0.75*H), int(0.25*W):int(0.75*W),:] = True
    mask[int(0.25*H):int(0.75*H), int(0.25*W):int(0.75*W),:] = 1
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
                                        #output_types=(tf.bool),
                                        output_types=(tf.float32),
                                        output_shapes=input_shape)
    # now a Dataset to load images
    img_ds = _image_file_dataset(filepaths, imshape=input_shape[:2], 
                                 num_channels=input_shape[2], norm=norm,
                                 num_parallel_calls=num_parallel_calls)
    img_ds = img_ds.shuffle(shuffle_queue)
    img_ds = img_ds.map(_augment, num_parallel_calls=num_parallel_calls)
    # combine the image and mask datasets
    zipped_ds = tf.data.Dataset.zip((img_ds, mask_ds))
    # precompute masked images for context encoder input
    masked_batched_ds = zipped_ds.batch(batch_size) #masked_img_ds.batch(batch_size)
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
    img_arr, mask
    """
    img_arr = np.stack([_load_img(f, norm=norm, num_channels=input_shape[2], 
                                  resize=input_shape[:2]) for f in filepaths])
    mask = _make_test_mask(*input_shape)
    mask = np.stack([mask for _ in range(img_arr.shape[0])])

    return img_arr, mask
    

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
    encoded = encoder(inpt)
    #inpt_mask = tf.keras.layers.Input(input_shape, name="inpt_mask")
    # Pathak's structure runs images through the encoder, then a dense
    # channel-wise layer, then dropout and a 1x1 Convolution before decoding.
    dense = ChannelWiseDense()(encoded)
    dropout = tf.keras.layers.Dropout(0.5)(dense)
    conv1d = tf.keras.layers.Conv2D(512,1)(dropout)
    decoded = decoder(conv1d)
    #encoded = encoder(inpt)
    #dense = ChannelWiseDense()(encoded)
    #dropout = tf.keras.layers.Dropout(0.5)(dense)
    #conv1d = tf.keras.layers.Conv2D(512,1)(dropout)
    #decoded = decoder(conv1d)
    # create a masked output to compare with ground truth (which should
    # already have it's unmasked areas set to 0)
    #masked_decoded = tf.keras.layers.Multiply(name="masked_decoded")(
    #                                [inpt_mask, decoded])
    
    # NOW FOR THE ADVERSARIAL PART
    if discriminator is None:
        discriminator = build_discriminator(input_shape[-1])
    #discriminator.compile(tf.keras.optimizers.Adam(0.1*learn_rate), 
    #                      loss=tf.keras.losses.binary_crossentropy)
    #discriminator.trainable = False
    #disc_pred = discriminator(decoded)
    
    #inpt = tf.keras.layers.Input((256,256,3))
    #encoded = encoder(inpt)
    #dense = ChannelWiseDense()(encoded)
    #dropout = tf.keras.layers.Dropout(0.5)(dense)
    #conv1d = tf.keras.layers.Conv2D(512,1)(dropout)
    #decoded = decoder(conv1d)
    inpainter = tf.keras.Model(inpt, decoded)
    

    #inpainter = tf.keras.Model([inpt, inpt_mask], 
    #                           [decoded, masked_decoded, disc_pred])
    #inpainter.compile(tf.keras.optimizers.Adam(learn_rate),
    #                  loss={"masked_decoded":tf.keras.losses.mse,
    #                        "discriminator":tf.keras.losses.binary_crossentropy},
    #                        loss_weights={"masked_decoded":1-disc_loss, 
    #                                      "discriminator":disc_loss})
    return inpainter, encoder, discriminator


def _stabilize(x):
    """
    Map values on the unit interval to [epsilon, 1-epsilon]
    """
    x = K.minimum(x, 1-K.epsilon())
    x = K.maximum(x, K.epsilon())
    return x


@tf.function
def inpainter_training_step(opt, inpainter, discriminator, img, mask, recon_weight=1, adv_weight=1e-3, clip_norm=0):
    """
    Tensorflow function for updating inpainter weights
    
    :opt: keras optimizer
    :inpainter: keras end-to-end context encoder model
    :discriminator: keras convolutional classifier to use as discriminator
    :img: batch of raw images
    :mask: batch of masks (1 in places to be removed, 0 elsewhere)
    :recon_weight: squared-error reconstruction loss weight
    :adv_weight: discriminator weight
    :clip_norm: if above 0, clip gradients to this norm
    
    Returns
    :reconstructed_loss: L2 norm loss for reconstruction
    :disc_loss: crossentropy loss from discriminator
    :total_loss: weighted sum of previous two
    """
    # inpainter update
    masked_img = (1-mask)*img
    with tf.GradientTape() as tape:
        # inpaint image
        inpainted_img = inpainter(masked_img)
        # compute difference between inpainted image and original
        reconstruction_residual = mask*(img - inpainted_img)
        reconstructed_loss = K.mean(K.square(reconstruction_residual))
        # compute adversarial loss
        disc_output_on_inpainted = discriminator(inpainted_img)
        #disc_loss_on_inpainted = K.sum(K.log(_stabilize(1-disc_output_on_inpainted)))
        # is the above line correct?
        disc_loss_on_inpainted = -1*K.mean(K.log(_stabilize(disc_output_on_inpainted)))
        # total loss
        total_loss = recon_weight*reconstructed_loss + adv_weight*disc_loss_on_inpainted
    
    variables = inpainter.trainable_variables
    gradients = tape.gradient(total_loss, variables)

    if clip_norm > 0:
        gradients = [tf.clip_by_norm(g, clip_norm) for g in gradients]
    
    opt.apply_gradients(zip(gradients, variables))
    
    return reconstructed_loss, disc_loss_on_inpainted, total_loss


@tf.function
def discriminator_training_step(opt, inpainter, discriminator, img, mask, clip_norm=0):
    """
    Tensorflow function for updating discriminator weights
    
    :opt: keras optimizer
    :inpainter: keras end-to-end context encoder model
    :discriminator: keras convolutional classifier to use as discriminator
    :img: batch of raw images
    :mask: batch of masks (1 in places to be removed, 0 elsewhere)
    :clip_norm: if above 0, clip gradients to this norm
    
    Returns
    """
    # inpainter update
    masked_img = (1-mask)*img
    with tf.GradientTape() as tape:
        # inpaint image
        inpainted_img = inpainter(masked_img)
        # compute adversarial loss
        disc_output_on_raw = discriminator(img) # try to get this close to zero
        disc_output_on_inpainted = discriminator(inpainted_img) # try to get this close to one
        
        disc_loss = -1*K.sum(K.log(_stabilize(disc_output_on_raw))) - \
                        K.sum(K.log(_stabilize(1-disc_output_on_inpainted)))
    
    variables = discriminator.trainable_variables
    gradients = tape.gradient(disc_loss, variables)

    if clip_norm > 0:
        gradients = [tf.clip_by_norm(g, clip_norm) for g in gradients]
    
    opt.apply_gradients(zip(gradients, variables))
    
    return disc_loss





def train_context_encoder(trainfiles, testfiles=None, inpainter=None, 
                          discriminator=None, num_epochs=1000, 
                          num_test_images=10, logdir=None, 
                          input_shape=(256,256,3), norm=255,
                          num_parallel_calls=4, batch_size=32,
                         recon_weight=1, adv_weight=1e-3, clip_norm=0, lr=1e-4):
    """
    Train your very own context encoder
    
    :trainfiles: list of paths to training files
    :testfiles: list of paths to test files
    :inpainter, discriminator: if not specified, will be auto-generated
    """
    inpaint_opt = tf.keras.optimizers.Adam(lr)
    disc_opt = tf.keras.optimizers.Adam(0.1*lr)
    
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
        test_ims, test_mask = _build_test_dataset(testfiles,
                                            input_shape=input_shape, norm=norm)
        test_masked_ims = (1-test_mask)*test_ims
        summary_writer = tf.contrib.summary.create_file_writer(logdir, 
                                                       flush_millis=10000)
        summary_writer.set_as_default()
        global_step = tf.compat.v1.train.get_or_create_global_step()

    else:
        test = False

    # combined training loop
    inpaint_step = True
    for e in range(num_epochs):
        # for each step in the epoch:
        for img, mask in train_ds:
            # prepare batch inputs
            #masked_img = masked_img.numpy()
            #img = img.numpy()
            #mask = mask.numpy()
            #target_img = target_img.numpy()
            # effective batch size
            #bs = img.shape[0]
            #ce_labels = np.ones(bs)
            #disc_labels = np.concatenate([
            #        np.zeros(bs),
            #        np.ones(bs)
            #])
            if inpaint_step:
                inpaint_losses = inpainter_training_step(inpaint_opt, inpainter, 
                                    discriminator, 
                                    img, mask, 
                                    recon_weight, adv_weight, clip_norm)
                inpaint_step = False
            else:
                disc_loss = discriminator_training_step(disc_opt, inpainter, discriminator, 
                                                img, mask, clip_norm)
                inpaint_step = True
            # run training step on inpainting network
            #inpainter.train_on_batch((masked_img, mask), 
            #                     (target_img, ce_labels))
            # generate reconstructed images
            #reconstructed_images = inpainter.predict((masked_img, mask))
            # make discriminator batch
            #disc_batch_x = np.concatenate([reconstructed_images[0], img], 0)
            # run discriminator training step
            #discriminator.train_on_batch(disc_batch_x, disc_labels)
    
        # at the end of the epoch, evaluate on test data.
        if test:
            
            test_ims, test_mask
            
            # evaluation- list of 3 values: ['loss', 'decoder_loss', 'discriminator_loss']
            #test_results = inpainter.evaluate(test_masked_ims, test_ims)
            # predict on the first few
            #preds = inpainter.predict(test_masked_ims[:num_test_images])
            preds = inpainter(test_masked_ims)
            # see how the discriminator does on them
            disc_outputs_on_raw = discriminator(test_ims)
            disc_outputs_on_inpaint = discriminator(preds)
            # for the visualization in tensorboard: replace the unmasked areas
            # with the input image as a guide to the eye
            preds = preds.numpy()[:num_test_images]*test_mask[:num_test_images] + \
                        test_ims[:num_test_images]*(1-test_mask[:num_test_images])
            predviz = np.concatenate([test_masked_ims[:num_test_images], preds], 
                             2).astype(np.float32)
            
            with tf.contrib.summary.always_record_summaries():
                tf.contrib.summary.scalar("total_loss", inpaint_losses[2], step=global_step)
                tf.contrib.summary.scalar("reconstruction_l2_loss", 
                                          inpaint_losses[0], step=global_step)
                tf.contrib.summary.scalar("discriminator_loss", 
                                          inpaint_losses[1], step=global_step)
                tf.contrib.summary.scalar("discriminator_training_loss", 
                                          disc_loss, step=global_step)
                tf.contrib.summary.histogram("discriminator_output_on_raw_images", 
                                         disc_outputs_on_raw, step=global_step)
                tf.contrib.summary.histogram("discriminator_output_on_inpainted", 
                                         disc_outputs_on_inpaint, step=global_step)
                for j in range(num_test_images):
                    tf.contrib.summary.image("img_%i"%j, 
                                     np.expand_dims(predviz[j,:,:,:],0), step=global_step)
        global_step.assign_add(1)
    return encoder, inpainter, discriminator




