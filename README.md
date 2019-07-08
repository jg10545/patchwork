# patchwork


### Interactive Machine Learning for an Imperfect World

This project is an experiment on how to leverage machine learning when the problem is poorly-specified. I'm interested in cases where we may not know *exactly* what we're looking for when we start the problem, but expect that starting to systematically sort through data would help us refine our question. In particular, we want a system for sorting through images that can handle:

* a set of classes that may be revised mid-project and may not be mutually disjoint (current approach: represent labels using multi-hot encoding)
* severe imbalances in one or more classes as well as in which classes are labeled (current approach: build training batches using stratified sampling)
* a relatively small number of labels (current approach: frozen feature extractor trained with self-supervised learning, active learning for motivating new images to label, few-shot models, semi-supervised loss function)
* partially-missing labels (current approach: masked multi-hot loss funtion)
* images whose properties may not be clear to the user(current approach: an option to exclude images during labeling)

If you're going to try this code out- I apologize in advance for the state of the GUI; I'm not really an interface guy.

Right now, 

* Labels are stored in a `pandas.DataFrame`
* The feature extractor and fine-tuning networks are `keras.Model` objects
* The interface is built with `panel`


`patchwork` has been tested with `tensorflow` 1.13.

* Free software: MIT license

## Installation

## Usage

To start with you'll need:

* A list of paths to all your image files
* The number of channels for each image
* A size to rescale all images to
* An initial (or revised) set of classes

## Label DataFrame

Labels are stored in a `pandas.DataFrame` containing:

* a `filepath` column containing the path to each image
* an `exclude` column (default `False`) indicating images to be excluded from the training set
* one column for each category with values `None`, `0`, or `1` (default `None`) indicating whether that image has that label (and if so, whether the class is present)

## Building Feature Extractors

### Context Encoder

```{python}
tf.enable_eager_execution()
# load paths to train and test files
trainfiles = [x.strip() for x in open("mytrainfiles.txt").readlines()]
testfiles = [x.strip() for x in open("mytestfiles.txt").readlines()]

# build inpainter, encoder, and discriminator networks
inpaint, encode, disc = patchwork.feature.build_inpainting_network(input_shape=(256,256,3))
# train
inpaint,disc = patchwork.feature.train_context_encoder(trainfiles,
                                        testfiles=testfiles,
                                        inpainter=inpaint,
                                        discriminator=disc,
                                        num_epochs=1000,
                                        logdir="logs/",
                                        batch_size=64,
                                        num_parallel_jobs=6)
```

### DeepCluster

*coming soon*
                                        
## Interactive Labeling and Fine-Tuning

More details forthcoming. Here are the basic steps to load the GUI inside a Jupyter notebook:

```{python}
import matplotlib.pyplot as plt
import panel as pn
pn.extension()
plt.ioff()

# prepare a DataFrame to hold labels if this is a new project (or
# just load the old DataFrame otherwise)
imfiles = [x.strip() for x in open("allmyfiles.txt").readlines()]                                     
classes = ["cat", "mammal", "dog"]
df = patchwork.prep_label_dataframe(imfiles, classes)

# load a feature extractor
fe = tf.keras.models.load_model("pretrained_feature_extractor.h5")
fe.trainable = False

# pass dataframe and feature extractor to a Patchwork object and
# load the GUI
pw = patchwork.Patchwork(df, feature_extractor=fe, imshape=(256,256), 
                        outfile="saved_labels.csv")
pw.panel()
```
                                        

## Credits

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
