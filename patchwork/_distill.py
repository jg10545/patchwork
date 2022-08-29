# -*- coding: utf-8 -*-
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from sklearn.metrics import accuracy_score, roc_auc_score

import patchwork as pw
from patchwork._losses import multilabel_distillation_loss
from patchwork._util import build_optimizer, compute_l2_loss
from patchwork.models import build_wide_resnet
from patchwork.feature._generic import GenericExtractor, _TENSORBOARD_DESCRIPTIONS

_fcn = {"vgg16":tf.keras.applications.VGG16,
        "vgg19":tf.keras.applications.VGG19,
        "resnet50":tf.keras.applications.ResNet50,
        "inception":tf.keras.applications.InceptionV3,
        "mobilenet":tf.keras.applications.MobileNetV2}


_DESCRIPTIONS = {
    "kl_loss":"Multilabel Kullback-Leibler divergence"
}
for d in _TENSORBOARD_DESCRIPTIONS:
    _DESCRIPTIONS[d] = _TENSORBOARD_DESCRIPTIONS[d]


def _build_student_model(model, output_dim, imshape=(256,256), num_channels=3):
    """
    Build a new student model or verify an existing one.

    Parameters
    ----------
    model : string or keras Model
        student model to use for distillation, or the name of a standard
        convnet design: vgg16, vgg19, resnet50, inception, or mobilenet
    output_dim : int
        Number of output dimensions (e.g. number of categories)
    imshape : tuple of ints, optional
        Image input shape. The default is (256,256).
    num_channels : int, optional
        Number of input image channels. The default is 3.

    Returns
    -------
    A keras Model object to use as the student
    """
    if isinstance(model, str):
        if model.lower().endswith(".h5"):
            model = tf.keras.models.load_model(model)
        else:
            if model.lower().startswith("wrn"):
                n = int(model.split("_")[1])
                k = int(model.split("_")[2])
                fcn = build_wide_resnet(n,k)
            else:
                assert model.lower() in _fcn, "I don't know what to do with model type %s"%model
                fcn = _fcn[model.lower()](weights=None, include_top=False)

            inpt = tf.keras.layers.Input((imshape[0], imshape[1], num_channels))
            net = fcn(inpt)
            net = tf.keras.layers.Flatten()(net)
            net = tf.keras.layers.Dense(output_dim, activation="sigmoid")(net)
            model = tf.keras.Model(inpt, net)
    else:
        assert isinstance(model, tf.keras.Model), "what is this model i dont even"
        assert model.output_shape[-1] == output_dim, "model output doesn't match output dimension"
    return model


def _build_distillation_training_function(student, opt, weight_decay=0, temp=1):
    def step_fn(x, y):
        lossdict = {}
        with tf.GradientTape() as tape:
            student_pred = student(x, training=True)
            lossdict["kl_loss"] = multilabel_distillation_loss(y, student_pred, temp=temp)
            loss = lossdict["kl_loss"]
            if weight_decay > 0:
                lossdict["l2_loss"] = compute_l2_loss(student)
                loss += weight_decay*lossdict["l2_loss"]
            lossdict["total_loss"] = loss
        gradients = tape.gradient(loss, student.trainable_variables)
        opt.apply_gradients(zip(gradients, student.trainable_variables))
        return lossdict
    return step_fn




class Distillerator(GenericExtractor):
    """
    Class for distilling a model using outputs of another model.
    """
    modelname = "Distillerator"

    def __init__(self, filepaths, ys, student,  testfiles=None, testlabels=None,
            lr=1e-3, opt_type="adam", lr_decay=0, decay_type="warmupcosine", temp=1,
            imshape=(256,256), num_channels=3, batch_size=128, norm=255, single_channel=False,
            num_parallel_calls=6, logdir=None, weight_decay=1e-6,
            class_names=None, strategy=None,  augment=False, notes="",
            **kwargs):
        """


        """
        output_dim = ys.shape[1]
        self.logdir = logdir
        self.filepaths = filepaths
        self.ys = ys
        self.strategy = strategy
        self._description = _DESCRIPTIONS
        if class_names is None:
            class_names = [str(i) for i in range(ys.shape[1])]
        self._class_names = class_names
        self._testlabels = testlabels

        if logdir is not None:
            self._file_writer = tf.summary.create_file_writer(logdir, flush_millis=10000)
            self._file_writer.set_as_default()

        # build model
        with self.scope():
            self._models = {"student":_build_student_model(student, output_dim,
                                   imshape, num_channels)}

        # SET UP THE INPUT PIPELINE
        ds, ns = pw.loaders.dataset(filepaths, ys=ys, imshape=imshape,
                                        num_channels=num_channels, shuffle=True,
                                        batch_size=batch_size, num_parallel_calls=num_parallel_calls,
                                        norm=norm, single_channel=single_channel, augment=augment)
        self._ds = self._distribute_dataset(ds)
        #self._ds = strategy.experimental_distribute_dataset(ds)

        # create optimizer
        self._optimizer = self._build_optimizer(lr, lr_decay, opt_type=opt_type,
                                                decay_type=decay_type,
                                                weight_decay=weight_decay)


        # build training function and distribute
        self._training_step = self._distribute_training_function(_build_distillation_training_function(self._models["student"],
                                                                                                       self._optimizer,
                                                                                                       weight_decay=weight_decay,
                                                                                                       temp=temp))
        # set up testing
        if (testfiles is not None) & (testlabels is not None):
            self._test_ds, self._test_ns = pw.loaders.dataset(testfiles, imshape=imshape,
                                                  num_channels=num_channels, shuffle=False,
                                                  batch_size=batch_size,
                                                  num_parallel_calls=num_parallel_calls,
                                                  augment=False)
            self._test = True
        else:
            self._test = False
        self.step = 0

        # parse and write out config YAML
        metrics = [f"auc_{c}" for c in class_names] + [f"acc_{c}" for c in class_names]
        self._parse_configs(augment=augment, temp=temp,
                            lr=lr, lr_decay=lr_decay,
                            imshape=imshape, num_channels=num_channels,
                            norm=norm, batch_size=batch_size,
                            num_parallel_calls=num_parallel_calls,
                            single_channel=single_channel, notes=notes,
                            trainer="distill", strategy=str(strategy),
                            weight_decay=weight_decay,
                            decay_type=decay_type, opt_type=opt_type,
                            metrics=metrics, **kwargs)

    def _run_training_epoch(self, **kwargs):
        """

        """
        for x, y in self._ds:
            lossdict = self._training_step(x,y)
            self._record_scalars(**lossdict)
            self._record_scalars(learning_rate=self._get_current_learning_rate())
            self.step += 1

    def evaluate(self, **kwargs):

        if self._test:
            predictions = self._models["student"].predict(self._test_ds, steps=self._test_ns)
            # compute performance metrics for each category
            # for i in range(output_dim):
            for e, c in enumerate(self._class_names):
                self._record_scalars(
                    **{f"auc_{c}":roc_auc_score(self._testlabels[:, e], predictions[:, e]),
                       f"acc_{c}":accuracy_score(self._testlabels[:, e],
                                                                 (predictions[:, e] >= 0.5).astype(int))}
                )

    def visualize_kernels(self, model=None):
        """
        Save a visualization to TensorBoard of all the kernels in the first
        convolutional layer
        """
        super(Distillerator).visualize_kernels(self._models["student"])

