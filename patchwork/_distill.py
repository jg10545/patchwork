# -*- coding: utf-8 -*-
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from sklearn.metrics import accuracy_score, roc_auc_score

import patchwork as pw
from patchwork._losses import multilabel_distillation_loss
from patchwork._util import build_optimizer
from patchwork.models import build_wide_resnet

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
        
    
    

def distill(filepaths, ys, student, epochs=5, testfiles=None, testlabels=None,
            lr=1e-3, opt_type="momentum", lr_decay=0, decay_type="exponential",
            imshape=(256,256), num_channels=3, batch_size=128,
            num_parallel_calls=6,
            class_names=None, 
            tracking_uri=None, experiment_name=None, 
            strategy=None, outfile=None, **kwargs):
    """
    Distill a student model from pre-computed teacher outputs.

    Parameters
    ----------
    filepaths : list of strings
        List of filepaths of images to train on
    ys : array
        Teacher outputs for each image. 1st dimension should be length of filepaths; second should be number of classes.
    student : string or Keras model
        Keras model to use as the student, or name of a model type to build (vgg16, vgg19, resnet50, inception, or mobilenet), or a WideResNet definition like "WRN_28_2", or path to saved model .h5
    testfiles : list of strings, optional
        List of filepaths of validation-set images
    testlabels : array, optional
        Array of ground truth labels, (len(testfiles), num_classes)
    epochs : int, optional
        Number of epochs to train
    lr : float, optional
        Learning rate. The default is 1e-3.
    opt_type : string, optional
        Which optimizer to train with- 'momentum' or 'adam'
    lr_decay: int, optional
        If set above 0, decay learning rate with this timescale. Set to -1 for
        one decay period by the end of training.
    decay_type: string, optional
        'exponential', 'staircase', or 'cosine'
    imshape : tuple of ints; optional
        Image input shape. The default is (256,256).
    num_channels : int, optional
        Number of input channels. The default is 3.
    batch_size : int
    num_parallel_calls : int
    class_names : list of strings; optional
        Names for each output category. If left blank, will use integers
    tracking_uri : string; optional
        URI for MLflow tracking server
    experiment_name : string; optional
        Name of MLflow experiment to log to
    strategy : tf.distribute.Strategy object; optional
        Distribution strategy for training on multiple GPUs
    outfile : str; optional
        Path to a location to save the model at the end of every epoch
    **kwargs : 
        Additional arguments passed to pw.loaders.dataset

    Returns
    -------
    student: tf.keras.Model
        The trained model
    :trainloss: list
        Training batch loss

    """
    if strategy is None:
        strategy = tf.distribute.get_strategy()
    if class_names is None:
        class_names = np.arange(ys.shape[1])
    if tracking_uri is not None:
        import mlflow, mlflow.keras
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        mlflow.log_params({"lr":lr, "opt_type":opt_type, "imshape":imshape,
                           "num_channels":num_channels, "lr_decay":lr_decay,
                           "decay_type":decay_type})
        if isinstance(student, str):
            mlflow.log_param("student", student)
    
    output_dim = ys.shape[1]
    outputs = {}
    with strategy.scope():
        # CREATE THE OPTIMIZER
        # SPECIAL CASE- we want to do one cycle of cosine decay
        if lr_decay < 0:
            lr_decay = int(epochs*len(filepaths)/batch_size)+1
        opt = build_optimizer(lr, lr_decay, opt_type, decay_type)
        # SET UP THE MODEL
        student = _build_student_model(student, output_dim,
                                   imshape, num_channels)
    
    # SET UP TESTING IF NECESSARY
    if (testfiles is not None)&(testlabels is not None):
        test_ds, test_ns = pw.loaders.dataset(testfiles, imshape=imshape,
                                num_channels=num_channels, shuffle=False,
                                batch_size=batch_size, 
                                num_parallel_calls=num_parallel_calls,
                                augment=False)
        #for c in range(output_dim):
        for c in class_names:
            outputs["auc_%s"%c] = []
            outputs["accuracy_%s"%c] = []
        
    
    # SET UP THE INPUT PIPELINE
    ds, ns = pw.loaders.dataset(filepaths, ys=ys, imshape=imshape,
                                num_channels=num_channels, shuffle=True,
                                batch_size=batch_size, num_parallel_calls=num_parallel_calls,
                                **kwargs)
    ds = strategy.experimental_distribute_dataset(ds)
        
    # CREATE A TRAINING FUNCTION
    def step_fn(x,y):
        with tf.GradientTape() as tape:
            student_pred = student(x, training=True)
            loss = multilabel_distillation_loss(y, student_pred, 1.)
        
        gradients = tape.gradient(loss, student.trainable_variables)
        opt.apply_gradients(zip(gradients, student.trainable_variables))
        return loss
    # AND DISTRIBUTE IF NECESSARY
    @tf.function
    def train_step(x,y):
        per_example_losses = strategy.run(step_fn, args=(x,y))
        return strategy.reduce(tf.distribute.ReduceOp.MEAN,
                               per_example_losses, axis=None)
    
    
    # TRAIN THE STUDENT MODEL
    train_loss = []
    step = 0
    for e in tqdm(range(epochs)):
        for x, y in ds:
            train_loss.append(train_step(x,y).numpy())
            step += 1
            
        # AT THE END OF EVERY EPOCH RUN TESTS
        if (testfiles is not None)&(testlabels is not None):
            predictions = student.predict(test_ds, steps=test_ns)
            # compute performance metrics for each category
            #for i in range(output_dim):
            for e,c in enumerate(class_names):
                auc = roc_auc_score(testlabels[:,e], predictions[:,e])
                acc = accuracy_score(testlabels[:,e], (predictions[:,e] >= 0.5).astype(int))
                outputs["auc_%s"%c].append(auc)
                outputs["accuracy_%s"%c].append(acc)
                if tracking_uri is not None:
                    mlflow.log_metrics({"auc_%s"%c:auc, "accuracy_%s"%c:acc},
                                       step=step)
        # AND SAVE A COPY OF THE MODEL
        if outfile is not None:
            student.save(outfile)
            
           
    outputs["train_loss"] = train_loss
    if tracking_uri is not None:
        mlflow.keras.log_model(student, "student_model")
        mlflow.end_run()
    return student, outputs
    