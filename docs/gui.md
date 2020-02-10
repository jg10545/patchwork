# Active Learning GUI

Here's the intended workflow:

* Start with a list of paths to all your image files
* Train a feature extractor for the images (as a `keras` model), or generate pre-extracted features (as a `numpy` array) for each image.
  * In the pre-extracted feature case, the rest of the workflow is lightweight enough that I can run it on an old laptop with no GPU
* Use `patchwork.prep_label_dataframe()` to initialize a `pandas` dataframe to store your labels (or start with an existing one). Since your time is precious, a careful representation of the time you spend annotating is the most important resource we'll build. The dataframe will have a few columns:
  * path to the file
  * a Boolean `validation` column (default `False`) that indicates the image should be used for testing any models we build and held out of training
  * a Boolean `exclude` column (default `False`) that records problematic images- record that you've seen it so you can find it again, but remove it from the training set.
  * one column per class that can take values `None` (unlabeled), `0` (doesn't contain class) or `1` (contains class). `patchwork` can handle partially-missing labels.
* Using your features and label dataframe, start a `patchwork.GUI` object.
  * Sample images to label
  * Select a fine-tuning model (maps generic feature tensors to task-specific feature vector) and output model (maps feature vector to probabilities over classes)
  * Train model and evaluate results
  * Lather, rinse, and repeat: use active learning to prioritize which images to label, and add capacity to your fine-tuning model as needed.
* When you've built a useful prototype, access the labels in `GUI.df` or the models in `GUI.models` to connect to the next step in whatever problem you're solving!

                                        
## Getting started

Here are the basic steps to load the GUI inside a Jupyter notebook:

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

# pass dataframe and feature extractor to a Patchwork object and
# load the GUI
gui = patchwork.GUI(df, feature_extractor=fe, imshape=(256,256), 
                        outfile="saved_labels.csv")
gui.panel()
```

You don't have to run inside Jupyter- check the [panel docs](https://panel.holoviz.org/user_guide/Deploy_and_Export.html) for instructions on how to deploy or embed the app elsewhere.

### Label tab

The Label tab is there you can annotate images. It has two main parts- a sampling tool and a labeling tool.

#### Sampling tool
                        
![](gui_label_sample.png)

Choose how to sample the next set of images- choose which subset of images to sample from (unlabeled, partially labeled, images containing a particular label, etc) and how to prioritize them. You'll probably sample differently as you build your dataset:

* For initial tagging- sort by `random`
* For active learning by uncertainty sampling- sort by `max entropy`
* For uncertainty sampling with respect to a particular class- sort by `maxent <class name>`
* For hard negative mining- sort by `high <class name>`

![](gui_label_classes.png)

### Model tab

![](gui_model.png)

### Train tab

![](gui_train.png)
