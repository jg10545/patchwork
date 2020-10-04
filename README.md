# patchwork


## Interactive Machine Learning for an Imperfect World

This project is an experiment on how to leverage machine learning for image classification problems, when the problem may be poorly-specified or evolving. I'm interested in cases where we may not know *exactly* what we're looking for when we start the problem, but expect that starting to systematically sort through data would help us refine our question. In particular, we want a system for sorting through images that can handle:

* a set of classes that may be revised mid-project and may not be mutually disjoint (current approach: represent labels using multi-hot encoding)
* severe imbalances in one or more classes as well as in which classes are labeled (current approach: build training batches using stratified sampling)
* a relatively small number of labels (current approach: combining a frozen feature extractor trained with active learning and few-shot methods; more below)
* partially-missing labels (current approach: masked multi-hot loss funtions)
* images that may have irreducible error with respect to the task at hand (current approach: an option to exclude images during labeling)
* uses standard data structures and modeling tools that can be scavenged and integrated into other tools and workflows (mostly `pandas` and `keras`).

If you're going to try this code out- I apologize in advance for the state of the GUI; I'm not really an interface guy. This library is a car with no seatbelts.


## What's inside

In the recent [SimCLRv2 paper](https://arxiv.org/abs/2006.10029), Chen *et al* lay out three steps for training a classifier without many labels: task-agnostic unsupervised pretraining of a feature extractor, task-specific supervised fine-tuning on only the labeled data, and finally task-specific semi-supervised learning on all the data. `patchwork` has tools for all three steps:

### Pre-training a feature extractor

The `patchwork.feature` module has [methods for pretraining convolutional networks as feature extractors](docs/feature.md).

* Self-supervised methods include context encoders, DeepCluster, MoCo, and SimCLR
* A [multitask learning](docs/multitask.md) interface for building feature extractors from whatever labeled data you have available. It automatically handles partially-missing labels (so you can combine disparate datasets with labels for different tasks) and can weigh tasks manually or automatically.
* All extractor trainers share a common set of [input and augmentation parameters](docs/input_aug.md)
* If you label a small number of patches before you start, all feature extractor trainers can automatically monitor linear model performance on a downstream task during training using TensorBoard
* SimCLR and multitask trainers should be able to train on multi-GPU systems. **Note that parts of the `tf.distribute` API are new and still experimental, so this may break in a `tensorflow` update**

### Training a supervised classifier

`patchwork` contains a  graphical user interface using the `panel` library for [interactive labeling](docs/gui.md). Using a frozen pre-trained feature extractor, iteratively label images, train a fine-tuning model to classify using your labels, then use the model to motivate which images to label next. Save out your classification model directly, or use the `pandas.DataFrame` of labels in your own workflow.

* The classifier shares [input and augmentation parameters](docs/input_aug.md) with the `feature` module.
* Active learning tools implemented include uncertainty sampling and GUIDE
* The GUI's modeling interface lets you mix and match different components and loss functions to add model capacity as your label set grows.

If you don't want to use my crappy GUI for training a supervised model, you're still welcome to scavenge any pieces you need- the loading functions used for the  GUI [can also be used with the Keras API](docs/input_keras.md).

### Semi-supervised fine tuning

This part's pretty unimaginative- starting with the model you trained using `patchwork.GUI` (or your own model) as a teacher, use `patchwork.distill()` to train a student model.

`patchwork` has been tested with `tensorflow` 2.0.

* Free software: MIT license



## Installation

use pip
                                        

## Credits

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

Images seen in my documentation are from the amazing [UC Merced Land Use dataset](http://weegee.vision.ucmerced.edu/datasets/landuse.html) which is wonderful for prototyping.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
