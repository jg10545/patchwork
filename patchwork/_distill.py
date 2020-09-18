# -*- coding: utf-8 -*-
from tqdm import tqdm
import tensorflow as tf
from sklearn.metrics import accuracy_score, roc_auc_score

import patchwork as pw
from patchwork._losses import multilabel_distillation_loss

_fcn = {"vgg16":tf.keras.applications.VGG16,
        "vgg19":tf.keras.applications.VGG19,
        "resnet50":tf.keras.applications.ResNet50V2,
        "inception":tf.keras.applications.InceptionV3,
        "mobilenet":tf.keras.applications.MobileNetV2}


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
        
    
    

def distill(filepaths, ys, student, epochs=5, testfiles=None, testlabels=None,
            lr=1e-3, optimizer="momentum",
            imshape=(256,256), num_channels=3, **kwargs):
    """
    

    Parameters
    ----------
    filepaths : list of strings
        List of filepaths of images to train on
    ys : array
        Teacher outputs for each image. 1st dimension should be length of filepaths; second should be number of classes.
    student : string or Keras model
        Keras model to use as the student, or name of a model type to build (vgg16, vgg19, resnet50, inception, or mobilenet)
    testfiles : list of strings, optional
        List of filepaths of validation-set images
    testlabels : array, optional
        Array of ground truth labels, (len(testfiles), num_classes)
    epochs : int, optional
        Number of epochs to train
    lr : float, optional
        Learning rate. The default is 1e-3.
    optimizer : string, optional
        Which optimizer to train with- 'momentum' or 'adam'
    imshape : tuple of ints; optional
        Image input shape. The default is (256,256).
    num_channels : int, optional
        Number of input channels. The default is 3.
    **kwargs : 
        Additional arguments passed to pw.loaders.dataset

    Returns
    -------
    student: tf.keras.Model
        The trained model
    :trainloss: list
        Training batch loss

    """
    output_dim = ys.shape[1]
    outputs = {}
    # CREATE THE OPTIMIZER
    if optimizer.lower() == "momentum":
        opt = tf.keras.optimizers.SGD(lr, momentum=0.9)
    elif optimizer.lower() == "adam":
        opt = tf.keras.optimizers.Adam(lr)
    else:
        assert False, "dont know what optimizer %s is"%optimizer
        
    # SET UP TESTING IF NECESSARY
    if (testfiles is not None)&(testlabels is not None):
        test_ds, test_ns = pw.loaders.dataset(testfiles, imshape=imshape,
                                num_channels=num_channels, shuffle=False,
                                augment=False)
        for c in range(output_dim):
            outputs["auc_%s"%c] = []
            outputs["accuracy_%s"%c] = []
        
    # SET UP THE MODEL
    student = _build_student_model(student, output_dim,
                                   imshape, num_channels)
    # SET UP THE INPUT PIPELINE
    ds, ns = pw.loaders.dataset(filepaths, ys=ys, imshape=imshape,
                                num_channels=num_channels, shuffle=True,
                                **kwargs)
        
    # CREATE A TRAINING FUNCTION
    @tf.function
    def train_step(x,y):
        with tf.GradientTape() as tape:
            student_pred = student(x, training=True)
            loss = multilabel_distillation_loss(y, student_pred, 1.)
        
        gradients = tape.gradient(loss, student.trainable_variables)
        opt.apply_gradients(zip(gradients, student.trainable_variables))
        return loss
    
    # TRAIN THE STUDENT MODEL
    train_loss = []
    for e in tqdm(range(epochs)):
        for x, y in ds:
            train_loss.append(train_step(x,y).numpy())
            
        # AT THE END OF EVERY EPOCH RUN TESTS
        if (testfiles is not None)&(testlabels is not None):
            predictions = student.predict(test_ds, steps=test_ns)
            # compute performance metrics for each category
            for i in range(output_dim):
                outputs["auc_%s"%i].append(roc_auc_score(testlabels[:,i], predictions[:,i]))
                outputs["accuracy_%s"%i].append(accuracy_score(testlabels[:,i], (predictions[:,i] >= 0.5).astype(int)))
            
           
    outputs["train_loss"] = train_loss
    return student, outputs
    