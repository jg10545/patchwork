# Multitask learning interface

The `MultiTaskTrainer` class manages multitask learning with hard parameter sharing. I recommend [Ruder's review](https://arxiv.org/abs/1706.05098) as a good place to start if you're interested.

* A fully-convolutional network that can be frozen or allowed to train
* Zero or more shared layers (if you're freezing the FCN then there should be some trainable shared layers)
* One head per task, with zero or more hidden layers




## Label format

The `MultiTaskTrainer` expects a DataFrame of labels that look something like this:

| filepaths | task1 | task2 | task3 |
| ----- | ----- | ----- | ----- | 
| file01.png | class1_a | class2_a | class3_a |
| file02.png | class1_b | `None` | class3_a |
| file03.png | `None` | class2_b | class3_a |

That is, one column for the filepaths and then one column per task. Partially-missing
labels are OK; by default the trainer stratifies sampling to try to balance missingness 
and the loss function is masked so that missing labels don't contribute to gradients
through the task heads.

The train and validation dataframes should have the same format.


## Specifying shared layer and task head structure

Everywhere that you can specify layers, the trainer inputs a list with one element 
per layer:
* **an integer:** add a convolutional layer with that many filters, the kernel size specified below, a ReLU activation and same pooling
* **"p":** add a 2x2 max pooling with 2x2 stride
* **"d":** add a 2D spatial dropout layer with rate specified below
* **"r":** add a convolutional residual block

For the task heads, the final output layer is added automatically by inferring
the number of classes from your training set.

## Task weights

Appropriately weighing the different tasks' weights to the loss function (so that
easy tasks don't wind up dominating) is tricky. I currently have three ways you 
can handle this:

* ignore the problem and hope it doesn't ruin your life (trainer applies equal weights)
* manually specify a list of weights using the `task_weights` keyword argument
* set `task_weights="adaptive"` to learn the weights using the method in Kendall *et al*'s [Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics](https://arxiv.org/abs/1705.07115)

## Example

```{python}
import pandas as pd
import tensorflow as tf
import patchwork as pw


# ---------- GET YOUR DATA TOGETHER ----------
train_labels = pd.read_csv("train_labels.csv")
val_labels = pd.read_csv("val_labes.csv")


# ---------- LOAD YOUR FEATURE EXTRACTOR ----------
fcn = tf.keras.models.load_model("my_pretrained_extractor.h5")

# ---------- CHOOSE AUGMENTATION PARAMETERS ----------
aug_params = {'gaussian_blur': 0.25,
             'flip_left_right': True,
             'flip_up_down': True,
             'rot90': True,
             'zoom_scale': 0.3}
             
# generally a good idea to visualize what your augmentation is doing
pw.viz.augplot(train_labels.trainfiles.values, aug_params)   


# ---------- TRAIN ----------
shared_layers = [512, "r"]
task_layers = [128, "p", "d", 256]
tasks = ["task_name_1", "task_name_2"] # column names in your dataframes
trainer = pw.feature.MultiTaskTrainer(logdir,
                            train_labels, val_labels,
                            tasks,
                            fcn,
                            shared_layers=shared_layers,
                            task_layers=task_layers,
                            task_weights="adaptive",
                            balance_probs=True,
                            augment=aug_params,
                            imshape=(256,256),
                            num_channels=3,
                            batch_size=64,
                            num_parallel_calls=6,
                            notes="initial multitask test")

trainer.fit(10)
```