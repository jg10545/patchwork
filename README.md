# patchwork


## Interactive Machine Learning for an Imperfect World

This project is an experiment on how to leverage machine learning when the problem is poorly-specified. I'm interested in cases where we may not know *exactly* what we're looking for when we start the problem, but expect that starting to systematically sort through data would help us refine our question. In particular, we want a system for sorting through images that can handle:

* a set of classes that may be revised mid-project and may not be mutually disjoint (current approach: represent labels using multi-hot encoding)
* severe imbalances in one or more classes as well as in which classes are labeled (current approach: build training batches using stratified sampling)
* a relatively small number of labels (current approach: frozen feature extractor trained with self-supervised learning, active learning for motivating new images to label, few-shot models, semi-supervised loss function)
* partially-missing labels (current approach: masked multi-hot loss funtions)
* images that may have irreducible error with respect to the task at hand (current approach: an option to exclude images during labeling)
* uses standard data structures and modeling tools that can be scavenged and integrated into other tools and workflows.

If you're going to try this code out- I apologize in advance for the state of the GUI; I'm not really an interface guy. This library is a car with no seatbelts.


### What's inside

`patchwork` has two main components (follow links for more details):

* A `feature` module for building [feature extractors](docs/feature.md) with self-supervision. Input a list of paths to unlabeled image files, and train a `keras` fully-convolutional network. 

* A graphical user interface using the `panel` library for [interactive labeling](docs/gui.md). Using a frozen pre-trained feature extractor, iteratively label images, train a fine-tuning model to classify using your labels, then use the model to motivate which images to label next. Save out your classification model directly, or use the `pandas.DataFrame` of labels in your own workflow.

Both parts of the library use a common set of [input and augmentation parameters](docs/input_aug.md).


`patchwork` has been tested with `tensorflow` 2.0.

* Free software: MIT license



## Installation

use pip
                                        

## Credits

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

Images seen in my documentation are from the amazing [UC Merced Land Use dataset](http://weegee.vision.ucmerced.edu/datasets/landuse.html) which is wonderful for prototyping.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
