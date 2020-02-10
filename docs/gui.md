# Active Learning GUI

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

### Label tab
                        
![](gui_label_sample.png)

![](gui_label_classes.png)

### Model tab

![](gui_model.png)

### Train tab

![](gui_train.png)