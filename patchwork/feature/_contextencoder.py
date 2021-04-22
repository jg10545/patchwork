"""

            Code for Pathak et al's Context Encoder

"""
import numpy as np
import tensorflow as tf
#import tensorflow.contrib
import tensorflow.keras.backend as K
import warnings

from patchwork._augment import augment_function
from patchwork.loaders import _image_file_dataset
from patchwork._util import _load_img
from patchwork._layers import ChannelWiseDense
from patchwork.feature._models import build_encoder, build_decoder, build_discriminator
from patchwork.feature._generic import GenericExtractor


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
    mask = np.zeros((H,W,C), dtype=np.float32)
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
                                   shuffle=True, num_parallel_calls=4,
                                   batch_size=32, prefetch=True, augment=False,
                                   single_channel=False):
    """
    Build a tf.data.Dataset object to use for training.
    """
    # first build a Dataset that generates masks
    def _gen():
        return mask_generator(*input_shape)
    mask_ds = tf.data.Dataset.from_generator(_gen,
                                        output_types=(tf.float32),
                                        output_shapes=input_shape)
    # now a Dataset to load images
    img_ds = _image_file_dataset(filepaths, imshape=input_shape[:2], 
                                 num_channels=input_shape[2], norm=norm,
                                 num_parallel_calls=num_parallel_calls,
                                 shuffle=True, single_channel=single_channel)

    if augment:
        _aug = augment_function(input_shape[:2], augment)
        #_aug = augment_function(augment)
        img_ds = img_ds.map(_aug, num_parallel_calls=num_parallel_calls)
    # combine the image and mask datasets
    zipped_ds = tf.data.Dataset.zip((img_ds, mask_ds))
    # precompute masked images for context encoder input
    masked_batched_ds = zipped_ds.batch(batch_size) 

    if prefetch:
        masked_batched_ds = masked_batched_ds.prefetch(1)
    return masked_batched_ds
    
    
def _build_test_dataset(filepaths, input_shape=(256,256,3), norm=255,
                        single_channel=False):
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
                                  resize=input_shape[:2]) 
                        for f in filepaths])

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
    # Pathak's structure runs images through the encoder, then a dense
    # channel-wise layer, then dropout and a 1x1 Convolution before decoding.
    dense = ChannelWiseDense()(encoded)
    dropout = tf.keras.layers.Dropout(0.5)(dense)
    conv1d = tf.keras.layers.Conv2D(512,1)(dropout)
    decoded = decoder(conv1d)
    
    # NOW FOR THE ADVERSARIAL PART
    if discriminator is None:
        discriminator = build_discriminator(input_shape[-1])

    inpainter = tf.keras.Model(inpt, decoded)

    return inpainter, encoder, discriminator


def _stabilize(x):
    """
    Map values on the unit interval to [epsilon, 1-epsilon]
    """
    x = K.minimum(x, 1-K.epsilon())
    x = K.maximum(x, K.epsilon())
    return x

def build_inpainter_training_step():
    @tf.function
    def inpainter_training_step(opt, inpainter, discriminator, img, mask, 
                                recon_weight=1, adv_weight=1e-3, imshape=(256,256)):
        
        """
        Tensorflow function for updating inpainter weights
    
        :opt: keras optimizer
        :inpainter: keras end-to-end context encoder model
        :discriminator: keras convolutional classifier to use as discriminator
        :img: batch of raw images
        :mask: batch of masks (1 in places to be removed, 0 elsewhere)
        :recon_weight: squared-error reconstruction loss weight
        :adv_weight: discriminator weight
        :imshape:
    
        Returns
        :reconstructed_loss: L2 norm loss for reconstruction
        :disc_loss: crossentropy loss from discriminator
        :total_loss: weighted sum of previous two
        """
        print("tracing inpainter training step")
        lossdict = {}
        # inpainter update
        masked_img = (1-mask)*img
        with tf.GradientTape() as tape:
            # inpaint image
            inpainted_img = inpainter(masked_img, training=True)[:,:imshape[0],:imshape[1],:]
            # compute difference between inpainted image and original
            reconstruction_residual = mask*(img - inpainted_img)
            reconstructed_loss = K.mean(K.abs(reconstruction_residual))
            # compute adversarial loss
            disc_output_on_inpainted = discriminator(inpainted_img)

            # is the above line correct?
            disc_loss_on_inpainted = -1*K.mean(K.log(_stabilize(disc_output_on_inpainted)))
            # total loss
            total_loss = recon_weight*reconstructed_loss + adv_weight*disc_loss_on_inpainted
            lossdict["inpainter_recon_loss"] = reconstructed_loss
            lossdict["inpainter_disc_loss"] = disc_loss_on_inpainted
            lossdict["inpainter_total_loss"] = total_loss

    
        variables = inpainter.trainable_variables
        gradients = tape.gradient(total_loss, variables)
    
        opt.apply_gradients(zip(gradients, variables))
    
        return lossdict
    return inpainter_training_step

def build_discriminator_training_step():
    @tf.function
    def discriminator_training_step(opt, inpainter, discriminator, img, mask):
        """
        Tensorflow function for updating discriminator weights
        
        :opt: keras optimizer
        :inpainter: keras end-to-end context encoder model
        :discriminator: keras convolutional classifier to use as discriminator
        :img: batch of raw images
        :mask: batch of masks (1 in places to be removed, 0 elsewhere)
    
        Returns discriminator loss
        """
        print("tracing discriminator training step")
        # inpainter update
        masked_img = (1-mask)*img
        with tf.GradientTape() as tape:
            # inpaint image
            inpainted_img = inpainter(masked_img)
            # compute adversarial loss
            disc_output_on_raw = discriminator(img, training=True) # try to get this close to zero
            disc_output_on_inpainted = discriminator(inpainted_img, training=True) # try to get this close to one
        
            disc_loss = -1*K.sum(K.log(_stabilize(disc_output_on_raw))) - \
                            K.sum(K.log(_stabilize(1-disc_output_on_inpainted)))
    
        variables = discriminator.trainable_variables
        gradients = tape.gradient(disc_loss, variables)
        opt.apply_gradients(zip(gradients, variables))
    
        return {"disc_loss":disc_loss}
    return discriminator_training_step
    


class ContextEncoderTrainer(GenericExtractor):
    """
    Class for training a context encoder.
    """
    modelname = "ContextEncoder"

    def __init__(self, logdir, trainingdata, testdata=None, fcn=None, inpainter=None,
                 discriminator=None, augment=True, 
                 recon_weight=1, adv_weight=1e-3, lr=1e-4,
                  imshape=(256,256), num_channels=3,
                 norm=255, batch_size=64, shuffle=True, num_parallel_calls=None,
                 single_channel=False, notes="",
                 downstream_labels=None):
        """
        :logdir: (string) path to log directory
        :trainingdata: (list or tf Dataset) list of paths to training images, or
            dataset to use for training loop
        :testdata: (list) filepaths of a batch of images to use for eval
        :fcn: (keras Model) fully-convolutional network to train as feature extractor
        :inpainter: (keras Model) full autoencoder for training
        :discriminator: (keras Model) discriminator for training
        :augment: (dict) dictionary of augmentation parameters, True for defaults or
            False to disable augmentation
        :recon_weight: (float) weight on reconstruction loss
        :adv_weight: (float) weight on adversarial loss
        :lr: (float) learning rate
        :imshape: (tuple) image dimensions in H,W
        :num_channels: (int) number of image channels
        :norm: (int or float) normalization constant for images (for rescaling to
               unit interval)
        :batch_size: (int) batch size for training
        :num_parallel_calls: (int) number of threads for loader mapping
        :single_channel: if True, expect a single-channel input image and 
            stack it num_channels times.
        :notes: (string) any notes on the experiment that you want saved in the
                config.yml file
        :downstream_labels: dictionary mapping image file paths to labels                
        """
        self.logdir = logdir
        self._downstream_labels = downstream_labels
        input_shape = (imshape[0], imshape[1], num_channels)
        
        self._file_writer = tf.summary.create_file_writer(logdir, flush_millis=10000)
        self._file_writer.set_as_default()
        
        if (fcn is None) or (inpainter is None) or (discriminator is None):
            inpainter, fcn, discriminator = build_inpainting_network(
                    input_shape=input_shape,
                    encoder=fcn)
        self.fcn = fcn
        self._models = {"fcn":fcn, "inpainter":inpainter, 
                        "discriminator":discriminator}
        
        # create optimizers
        self._optimizer = {
                "inpaint":tf.keras.optimizers.Adam(lr),
                "disc":tf.keras.optimizers.Adam(0.1*lr)
                }
        
        self._inpainter_training_step = build_inpainter_training_step()
        self._discriminator_training_step = build_discriminator_training_step()
        
        # build training dataset
        if isinstance(trainingdata, list):
            self._train_ds = _build_context_encoder_dataset(trainingdata, 
                                        input_shape=input_shape, 
                                norm=norm, shuffle=True, 
                                num_parallel_calls=num_parallel_calls,
                                batch_size=batch_size, prefetch=True,
                                augment=augment, 
                                single_channel=single_channel)
        else:
            assert isinstance(trainingdata, tf.data.Dataset), "i don't know what to do with this"
            self._train_ds = trainingdata
        
        # build evaluation dataset
        if testdata is not None:
            self._test_ims, self._test_mask = _build_test_dataset(testdata,
                                            input_shape=input_shape,
                                            norm=norm, 
                                            single_channel=single_channel)
            self._test_masked_ims = (1-self._test_mask)*self._test_ims
            self._test = True
        else:
            self._test = False
            
        self.step = 0
        
        # let's do a quick check on the model- it's possible to define an
        # encoder/decoder pair that won't return the same shape as the input 
        # shape. not NECESSARILY a problem but the user should know.
        output_shape = self._models["inpainter"].output_shape[1:3]
        if output_shape != imshape:
            warnings.warn("imshape %s and context encoder output shape %s are different. ye be warned."%(imshape, output_shape))
        
        # parse and write out config YAML
        self._parse_configs(augment=augment, recon_weight=recon_weight,
                            adv_weight=adv_weight, lr=lr,
                            imshape=imshape, num_channels=num_channels,
                            norm=norm, batch_size=batch_size, 
                            num_parallel_calls=num_parallel_calls, 
                            single_channel=single_channel, notes=notes,
                            trainer="contextencoder")
        
        
    def _run_training_epoch(self, **kwargs):
        """
        
        """
        for img, mask in self._train_ds:
            # alternatve between inpainter and discriminator training
            if self.step % 2 == 0:
                lossdict = self._inpainter_training_step(
                        self._optimizer["inpaint"], 
                        self._models["inpainter"],
                        self._models["discriminator"],
                        img, mask,
                        recon_weight=self.config["recon_weight"],
                        adv_weight=self.config["adv_weight"],
                        imshape=self.input_config["imshape"])
                self._record_scalars(**lossdict)
            else:
                disc_loss = self._discriminator_training_step(
                                self._optimizer["disc"], 
                                self._models["inpainter"], 
                                self._models["discriminator"],
                                img, mask)
                self._record_scalars(**disc_loss)
            self.step += 1

           
    def evaluate(self, avpool=True, query_fig=False):
        num_test_images=10
        if self._test:
            preds = self._models["inpainter"].predict(self._test_masked_ims)
            preds = preds[:,:self.input_config["imshape"][0], :self.input_config["imshape"][1],:]
            
            reconstruction_residual = self._test_mask*(preds - self._test_ims)
            reconstructed_loss = np.mean(np.abs(reconstruction_residual))
            
            # see how the discriminator does on them
            disc_outputs_on_raw = self._models["discriminator"].predict(self._test_ims)
            disc_outputs_on_inpaint = self._models["discriminator"].predict(preds)
            # for the visualization in tensorboard: replace the unmasked areas
            # with the input image as a guide to the eye
            preds = preds*self._test_mask + self._test_ims*(1-self._test_mask)
            predviz = np.concatenate([self._test_masked_ims[:num_test_images], 
                                      preds[:num_test_images]], 
                             2).astype(np.float32)

            # record all the summaries
            tf.summary.image("inpaints", predviz, step=self.step, max_outputs=10)
            tf.summary.histogram("disc_outputs_on_raw", disc_outputs_on_raw,
                                 step=self.step)
            tf.summary.histogram("disc_outputs_on_inpaint", disc_outputs_on_inpaint,
                                 step=self.step)
            self._record_scalars(test_recon_loss=reconstructed_loss)
            
        if self._downstream_labels is not None:
            # choose the hyperparameters to record
            if not hasattr(self, "_hparams_config"):
                from tensorboard.plugins.hparams import api as hp
                hparams = {
                    hp.HParam("adv_weight", hp.RealInterval(0., 10000.)):self.config["adv_weight"]
                    }
            else:
                hparams=None
            self._linear_classification_test(hparams, avpool=avpool, query_fig=query_fig)
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        