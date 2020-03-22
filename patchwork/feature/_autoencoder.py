import numpy as np
import tensorflow as tf

from patchwork.loaders import dataset
from patchwork.feature._generic import GenericExtractor


def _build_autoencoder(num_channels=3, conv_layers=[32, 48, 64, 128], 
                       dropout=0.5):
    """
    
    """
    # encoder
    inpt = tf.keras.layers.Input((None, None, 3))
    net = inpt

    for c in conv_layers:
        net = tf.keras.layers.Conv2D(c, 3, activation="relu",
                                strides=(2,2), padding="same")(net)
    encoder = tf.keras.Model(inpt, net)
    
    # decoder
    decoder_inpt = tf.keras.Input((None, None, conv_layers[-1]))
    net = decoder_inpt

    for c in conv_layers[::-1]:
        net = tf.keras.layers.Conv2DTranspose(c, 3,
                                         padding="same",
                                         activation="relu",
                                         strides=(2,2))(net)
    net = tf.keras.layers.Conv2D(3, 3, padding="same",
                            activation="sigmoid")(net)
    decoder = tf.keras.Model(decoder_inpt, net)
    
    # full_model
    inpt_full = tf.keras.layers.Input((256,256,3))
    encoded = encoder(inpt_full)
    if dropout > 0:
        encoded = tf.keras.layers.SpatialDropout2D(0.5)(encoded)
    decoded = decoder(encoded)
    full_model = tf.keras.Model(inpt_full, decoded)
    
    return encoder, full_model
    
    

def _build_training_step(model, opt):
    variables = model.trainable_variables
    @tf.function
    def training_step(x):
        with tf.GradientTape() as tape:
            reconstructed = model(x, training=True)
            loss = tf.reduce_mean(
                    tf.keras.losses.mae(x, reconstructed)
            )
        gradient = tape.gradient(loss, variables)
        opt.apply_gradients(zip(gradient, variables))
        return loss        
    return training_step





class AutoEncoderTrainer(GenericExtractor):
    """
    Generic convolutional autoencoder
    """

    def __init__(self, logdir, trainingdata, testdata=None, fcn=None, full_model=None,
                 conv_layers=[32, 48, 64, 128], dropout=0.5,
                 augment=True, 
                 lr=1e-3, lr_decay=100000,
                 imshape=(256,256), num_channels=3,
                 norm=255, batch_size=64, num_parallel_calls=None,
                 single_channel=False, notes="",
                 downstream_labels=None):
        """
        :logdir: (string) path to log directory
        :trainingdata: (list) list of paths to training images
        :testdata: (list) filepaths of a batch of images to use for eval
        :fcn: (keras Model) fully-convolutional network to train as feature extractor
        :full_model: (keras model) full autoencoder
        :augment: (dict) dictionary of augmentation parameters, True for defaults or
                False to disable augmentation
        :lr: (float) initial learning rate
        :lr_decay: (int) steps for learning rate to decay by half (0 to disable)
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
        self.trainingdata = trainingdata
        self._downstream_labels = downstream_labels
        
        self._file_writer = tf.summary.create_file_writer(logdir, flush_millis=10000)
        self._file_writer.set_as_default()
        
        # build models if necessary
        if fcn is None or full_model is None:
            print("building new autoencoder")
            fcn, full_model = _build_autoencoder(num_channels, conv_layers, dropout)
        self.fcn = fcn
        self._models = {"fcn":fcn, "full":full_model}    
        
        # create optimizer
        if lr_decay > 0:
            learnrate = tf.keras.optimizers.schedules.ExponentialDecay(lr, 
                                            decay_steps=lr_decay, decay_rate=0.5,
                                            staircase=False)
        else:
            learnrate = lr
        self._optimizer = tf.keras.optimizers.Adam(learnrate)
        
        # training dataset
        self._train_ds, _ = dataset(trainingdata, imshape=imshape,norm=norm,
                                    sobel=False, num_channels=num_channels,
                                    augment=augment, single_channel=single_channel,
                                    batch_size=batch_size, shuffle=True)
        # build evaluation dataset
        if testdata is not None:
            self._test_ds, self._test_steps = dataset(testdata,
                                     imshape=imshape,norm=norm,
                                     sobel=False, num_channels=num_channels,
                                     single_channel=single_channel,
                                     batch_size=batch_size, shuffle=False)
            self._test = True
        else:
            self._test = False
        
        
        # build training step function- this step makes sure that this object
        # has its own @tf.function-decorated training function so if we have
        # multiple deepcluster objects (for example, for hyperparameter tuning)
        # they won't interfere with each other.
        self._training_step = _build_training_step(full_model, self._optimizer)
        self.step = 0
        
        # parse and write out config YAML
        self._parse_configs(augment=augment, conv_layers=conv_layers,
                            dropout=dropout, lr=lr, 
                            lr_decay=lr_decay,
                            imshape=imshape, num_channels=num_channels,
                            norm=norm, batch_size=batch_size,
                            num_parallel_calls=num_parallel_calls, 
                            single_channel=single_channel, notes=notes)
        
        
    def _run_training_epoch(self, **kwargs):
        """
        
        """
        for x in self._train_ds:
            loss = self._training_step(x)
            self._record_scalars(reconstruction_loss=loss)
            self.step += 1
 
    def evaluate(self):
        if self._test:
            test_recon_loss = 0
            for x in self._test_ds:
                reconstructed = self._models["full"](x)
                test_recon_loss += np.mean(np.abs(x.numpy()-reconstructed.numpy()))
                
            self._record_scalars(test_reconstruction_loss=test_recon_loss/self._test_steps)
            
            test_ims = np.concatenate([x.numpy(), reconstructed.numpy()], 2)
            self._record_images(test_images=test_ims)
                    
        if self._downstream_labels is not None:
            # choose the hyperparameters to record
            if not hasattr(self, "_hparams_config"):
                from tensorboard.plugins.hparams import api as hp
                hparams = {
                    hp.HParam("dropout", hp.RealInterval(0., 1.)):self.config["dropout"]
                    }
            else:
                hparams=None
            self._linear_classification_test(hparams)
            
            
            