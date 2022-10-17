import numpy as np
import tensorflow as tf
import sklearn.preprocessing

from patchwork.loaders import _image_file_dataset
from patchwork._tfrecord import save_dataset_to_tfrecords

from patchwork.feature._text_transformer import build_text_transformer
from patchwork.feature._simclr import _gather
from patchwork.feature._generic import GenericExtractor

def _wrap_encoder(encoder, maxlen, prompt_func=None, tfpyfunc=True):
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

    if tfpyfunc:
        # then wrap that as a tf.py_func so autograph doesn't get
        # in here and start breaking stuff
        def prompt(x, y):
            return tf.py_function(_wrap, inp=[x], Tout=tf.int64), y
        return prompt
    else:
        return _wrap

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


def save_lit_dataset(prompts, filepaths, fcn, outdir, imshape=(256, 256), num_parallel_calls=-1, norm=255,
                     num_channels=3, shuffle=True, single_channel=False,
                     augment=False, batch_size=32, num_shards=10, gzip=True):
    """

    """
    # wrap fcn with a pooling layer so we get a single vector per image
    if single_channel:
        inpt = tf.keras.layers.Input(imshape)
    else:
        inpt = tf.keras.layers.Input((imshape[0], imshape[1], num_channels))
    net = fcn(inpt)
    net = tf.keras.layers.GlobalAveragePooling2D()(net)
    model = tf.keras.Model(inpt, net)
    # load the images- dataset will be (image, prompt)
    load_ds = _image_file_dataset(filepaths, ys=prompts, imshape=imshape,
                                        num_parallel_calls=num_parallel_calls, norm=norm,
                                        num_channels=num_channels, shuffle=shuffle,
                                        single_channel=single_channel, augment=augment)
    load_ds = load_ds.batch(batch_size)
    # run each image through the feature extractor and swap order- dataset will be batches of (prompt, feature)
    def _gen():
        for x,y in load_ds:
            yield y, model(x, training=False)
    ds = tf.data.Dataset.from_generator(_gen, output_types=(tf.string, tf.float32))
    # unbatch so we get single instances of (prompt, feature)
    ds = ds.unbatch()
    # write to file
    save_dataset_to_tfrecords(ds, outdir, num_shards=num_shards, gzip=gzip)


def load_lit_dataset_from_tfrecords(record_dir, d, shuffle=2048, num_parallel_calls=-1,
                                    map_fn=None, gzip=True):
    """
    Load a directory structure of tfrecord files (like you'd build with save_to_tfrecords)
    into a tensorflow dataset for training a LiT model.

    Assumes tfrecord files are saved with .tfrecord or .snapshot.

    :record_dir: top-level directory containing record files
    :d: embedding dimension for image features
    :shuffle: if not False, size of shuffle queue
    :num_parallel_calls: number of parallel readers/mappers for loading and parsing
        the dataset
    :map_fn: function to map across dataset during loading
    :gzip: whether tfrecord was saved using GZIP compression
    """
    if gzip:
        comp = "GZIP"
    else:
        comp = "NONE"

    element_spec = (
        tf.TensorSpec(shape=(), dtype=tf.string),
        tf.TensorSpec(shape=(d), dtype=tf.float32)
    )
    # note that this function may change in the future
    ds = tf.data.experimental.load(record_dir, element_spec, compression=comp)
    # if a map function was included, map across the dataset
    if map_fn:
        ds = ds.map(map_fn, num_parallel_calls=num_parallel_calls)
    if shuffle:
        ds = ds.shuffle(shuffle)
    return ds



def build_lit_training_step(text_model, optimizer, temp=0.07, weight_decay=0):
    # check whether we're in mixed-precision mode
    mixed = tf.keras.mixed_precision.global_policy().name == 'mixed_float16'
    def trainstep(text_batch, img_embedding_batch):
        img_embedding_batch = tf.nn.l2_normalize(img_embedding_batch, 1)
        z1 = _gather(img_embedding_batch)
        labels = tf.arange(text_batch.shape[0])
        with tf.GradientTape() as tape:
            text_embed = text_model(text_batch, training=True)
            text_embed = tf.nn.l2_normalize(text_embed, 1)
            z2 = _gather(text_embed)

            logits1 = tf.matmul(z1, z2, transpose_b=True) / temp
            logits2 = tf.matmul(z2, z1, transpose_b=True) / temp

            prediction1 = tf.argmax(logits1, axis=1)
            prediction2 = tf.argmax(logits2, axis=1)
            acc1 = tf.reduce_mean(tf.cast(prediction1 == labels, tf.float32))
            acc2 = tf.reduce_mean(tf.cast(prediction2 == labels, tf.float32))
            nce_batch_acc = 0.5*(acc1 + acc2)

            loss1 = tf.reduce_mean(
                tf.losses.sparse_categorical_crossentropy(labels, logits1, from_logits=True))
            loss2 = tf.reduce_mean(
                tf.losses.sparse_categorical_crossentropy(labels, logits2, from_logits=True))
            xent_loss = 0.5*(loss1 + loss2)

            if (weight_decay > 0)&("LARS" not in optimizer._name):
                l2_loss = compute_l2_loss(text_model)
            else:
                l2_loss = 0

            loss = nce_loss + weight_decay*l2_loss
            if mixed:
                loss = optimizer.get_scaled_loss(loss)
        grads = tape.gradient(loss, textmodel.trainable_variables)
        optimizer.apply_gradients(zip(grads, textmodel.trainable_variables))
        return {"nt_xent_loss": xent_loss,
                "l2_loss": l2_loss,
                "loss": loss,
                "nce_batch_acc": nce_batch_acc}
        return lossdict
    return trainstep


def _zero_shot_accuracy_test(image_features, prompts, encoder, text_model, maxlen=72):
    """
    :image_features: (N,d) numpy array of image features
    :prompts: length-N list of strings; prompt for each image
    :encoder: trained SentencePiece encoder object
    :text_model: Keras model; text encoder
    """
    # label encoder- convert strings to an integer index
    labelenc = sklearn.preprocessing.OneHotEncoder()
    prompt_labels = np.array(labelenc.fit_transform(prompts).argmax(1)).ravel()

    # get normalized encodings for each unique prompt
    enc = _wrap_encoder(encoder, maxlen, tfpyfunc=False)
    prompt_encodings = np.array([enc(x) for x in labelenc.categories_[0]])
    prompt_features = text_model.predict(prompt_encodings)
    prompt_features = sklearn.preprocessing.normalize(prompt_features, norm='l2', axis=1)
    # normalize image encodings
    image_features = sklearn.preprocessing.normalize(image_features, norm='l2', axis=1)

    # for each image find the highest dot-product to a prompt category
    highest_dot_products = image_features.dot(prompt_features.T).argmax(1)

    return np.mean(prompt_labels == highest_dot_products)

class LiTTrainer(GenericExtractor):
    """
    Class for training a LiT model.

    Based on "LiT: Zero-Shot Transfer with Locked-image text Tuning" by Zhai et al.
    """
    modelname = "LiT"

    def __init__(self, logdir, text_model, tokenizer, trainingdata,
                 testdata=None, testlabels=None,
                 maxlen=76, #embed_dim=512, ff_dim=2048,
                 #num_layers=12, num_heads=8,
                 temperature=0.07, #output_dim=64,
                 weight_decay=0,
                 lr=0.01, lr_decay=0, decay_type="cosine",
                 opt_type="adam",
                 prompt_func=None,
                 zero_shot_tests=None,
                 #imshape=(256,256), num_channels=3,
                 #norm=255,
                 batch_size=64, num_parallel_calls=None,
                 #single_channel=False,
                 notes="",
                 #downstream_labels=None,
                 strategy=None, **kwargs):
        """
        :logdir: (string) path to log directory
        :tokenizer: (string) path to sentencepiece model file OR a custom tokenizer
        :trainingdata: (list) list of strings; paths to training images
        :traininglabels: (list) list of strings; captions for training images
        :testdata: (list) filepaths of a batch of images to use for eval
        :testlabels: (list) list of strings; captions for test images
        :fcn: (keras Model) fully-convolutional network to train as feature extractor
        :augment: (dict) dictionary of augmentation parameters, True for defaults
        :maxlen: int; length to pad or truncate encoded text queries to
        :embed_dim: int; embedding dimension of tokens and transformers for language model
        :ff_dim: int; dimension of internal feed-forward layers inside transformer blocks
        :num_layers: int; number of transformer blocks in language model
        :num_heads: int; number of heads in each transformer block in language model
        :temperature: the Boltzmann temperature parameter- rescale the cosine similarities by this factor before computing softmax loss.
        :output_dim: dimension of projection head's output space.
        :weight_decay: coefficient for L2-norm loss. The original SimCLR paper used 1e-6.
        :lr: (float) initial learning rate
        :lr_decay:  (int) number of steps for one decay period (0 to disable)
        :decay_type: (string) how to decay the learning rate- "exponential" (smooth exponential decay), "staircase" (non-smooth exponential decay), or "cosine"
        :opt_type: (string) optimizer type; "adam" or "momentum"
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
        :strategy: if distributing across multiple GPUs, pass a tf.distribute
            Strategy object here
        """
        output_dim = text_model.output_shape[-1]
        self.logdir = logdir
        self.trainingdata = trainingdata
        self._downstream_labels = downstream_labels
        self._test_index_updated = False
        if strategy is None:
            strategy = tf.distribute.get_strategy()
        self.strategy = strategy
        self._zero_shot_tests = zero_shot_tests


        self._file_writer = tf.summary.create_file_writer(logdir, flush_millis=10000)
        self._file_writer.set_as_default()
        # load tokenizer
        if isinstance(tokenizer, str):
            self._tokenizer = spm.SentencePieceProcessor(tokenizer)
        else:
            self._tokenizer = tokenizer
        self._vocab_size = self._tokenizer.vocab_size()

        self._models = {"text":text_model}

        # build training dataset
        # if user passes location of training tfrecords- load as a dataset
        if isinstance(trainingdata, str):
            ds = load_lit_dataset_from_tfrecords(trainingdata, embed_dim,
                                                       num_parallel_calls=num_parallel_calls,
                                                       map_fn=_wrap_encoder(encoder, maxlen,
                                                                            prompt_func=prompt_func))
        else:
            ds = trainingdata
        self._ds = self._distribute_dataset(ds)

        # create optimizer
        self._optimizer = self._build_optimizer(lr, lr_decay, opt_type=opt_type,
                                                decay_type=decay_type,
                                                weight_decay=weight_decay)


        # build training step
        trainstep = build_lit_training_step(text_model, self._optimizer,
                                            temp=temperature, weight_decay=weight_decay)
        self._training_step = self._distribute_training_function(trainstep)
        """
        if testdata is not None:
            self._test_ds = clip_dataset(testdata, testlabels, self._tokenizer,
                                         maxlen=maxlen, imshape=imshape,
                                         num_channels=num_channels,
                                         num_parallel_calls=num_parallel_calls,
                                         norm=norm, batch_size=batch_size, shuffle=False)
            @tf.function
            def loss_step(x,y):
                img_embed = self._models["full"](x, training=False)
                text_embed = self._models["text"](y, training=False)
                img_norm = tf.nn.l2_normalize(img_embed, 1)
                text_norm = tf.nn.l2_normalize(text_embed, 1)
                return _contrastive_loss(img_norm, text_norm, temperature)
            self._loss_step = loss_step
            self._test = True
        else:
            self._test = False"""

        self.step = 0

        # parse and write out config YAML
        metrics= ["linear_classification_accuracy",
                   "test_acc"]
        self._parse_configs(metrics=metrics,
                            tokenizer=tokenizer, maxlen=maxlen,
                            #augment=augment,
                            temperature=temperature,
                            output_dim=output_dim, weight_decay=weight_decay,
                            #num_layers=num_layers,
                            #num_heads=num_heads,
                            lr=lr, lr_decay=lr_decay,
                            #imshape=imshape, num_channels=num_channels,
                            #norm=norm,
                            # batch_size=batch_size,
                            num_parallel_calls=num_parallel_calls,
                            #single_channel=single_channel,
                            # notes=notes,
                            trainer="lit", strategy=str(strategy),
                            decay_type=decay_type, opt_type=opt_type, **kwargs)

    def _run_training_epoch(self, **kwargs):
        """

        """
        self._test_index_updated = False
        for x, y in self._ds:
            lossdict = self._training_step(x, y)
            self._record_scalars(**lossdict)
            self._record_scalars(learning_rate=self._get_current_learning_rate())
            self.step += 1

    def _run_zero_shot_tests(self, zero_shot_tests, suffix=""):
        # if user passes an (image features, prompts) tuple: run zero-shot accuracy
        if isinstance(zero_shot_tests, tuple):
            z = zero_shot_tests
            acc = _zero_shot_accuracy_test(z[0], z[1], self._tokenizer,
                                           self._models["text"], maxlen=self.config["maxlen"])
            key = "zero_shot_accuracy"
            if len(suffix) > 0:
                key += "_" + suffix
            self._record_scalars(**{key:acc})
        # if user passed a dict of (image features, prompts) tuples- run this function once
        # for each dataset
        elif isinstance(zero_shot_tests, dict):
            for k in dict:
                self._run_zero_shot_tests(zero_shot_tests[k], suffix=k)


    def evaluate(self, *args, **kwargs):
        if self._zero_shot_tests is not None:
            self._run_zero_shot_tests(self._zero_shot_tests)

    def load_weights(self, logdir):
        """
        Update model weights from a previously trained model

        Different from generic load_weights because we're using TF
        savedmodel format instead of HDF5
        """
        super().load_weights(logdir)
        #for k in self._models:
        #    savedloc = os.path.join(logdir, k, "variables", "variables")
        #    self._models[k].load_weights(savedloc)
        self._test_index_updated = False


    def _update_test_index(self):
        """
        Precompute embeddings on the test set so we can run sample
        queries against it.
        """
        if not self._test_index_updated:
            logging.info("updating embedding index on test data")
            self._test_index = self._models["full"].predict(self._test_ds)
            self._test_index_updated = True


    def query_test_set(self, querystring, plot=True):
        """

        """
        maxlen = self.config["maxlen"]
        # make sure test index is up-to-date
        self._update_test_index()
        # run byte-pair encoding of query text
        query_array = np.array(self._tokenizer.encode(querystring,
                                                      out_type=int, add_bos=True,
                                                      add_eos=True))
        # pad or clip query as necessary
        L = len(query_array)
        if L > maxlen:
            query_array = query_array[:maxlen]
        elif L < maxlen:
            query_array = np.concatenate([query_array,
                                          np.zeros(maxlen-L, dtype=np.int64)])
        query_array = query_array.reshape(1,-1)

        # now map that query to a semantic vector in the shared
        # embedding space
        query_vector = self._models["text"](query_array, training=False)
        # find distance to all the image embeddings from the test set
        dists = cdist(self._test_index, query_vector.numpy(), metric="cosine").ravel()
        ordering = dists.argsort()

        if plot:
            plt.suptitle("Query: '%s'"%querystring, fontsize=14)
            for i in range(9):
                plt.subplot(3,3,i+1)
                img = Image.open(self._testdata[ordering[i]])
                plt.imshow(img)
                plt.axis(False)
                plt.title(self._testlabels[ordering[i]].replace(".",".\n").replace(",",",\n"))
        else:
            return dists

