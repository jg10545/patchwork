# Feature Extractor trainers

The `patchwork.feature` module has several models implemented for unsupervised or self-supervised representation learning:

* [Context Encoders](https://arxiv.org/abs/1604.07379)
* [DeepCluster](https://arxiv.org/abs/1807.05520)
* [SimCLR](https://arxiv.org/abs/2002.05709) (along with multi-GPU version)
* [MoCo](https://arxiv.org/abs/1911.05722)

The module has a class to manage the training of each model. You can initialize the trainer with a fully-convolutional `keras` model for it to train (otherwise it will use a default model). The trainers are meant to be as similar as possible, to make it easy to throw different self-supervision approches at your problem and see how they compare. Here's what they have in common:

* Each feature extractor training class shares common [input pipeline and augmentation parameters](input_aug.md). A copy of your parameters is saved in a YAML file in the log directory for reproducibility.
* Choose between `adam` and `momentum` optimizers, with no learning rate decay, or `exponential` (smooth exponential decay), `staircase` (LR cut in half every `lr_decay` steps), or `cosine` decay.
* Call `trainer.load_weights()` to import weights of all the training components from a previous run
* Input a list of paths to test files to compute out-of-sample metrics and visualizations for TensorBoard (particularly useful for context encoders)
* If you have a small number of labels for a downstream task, the trainer will automate the linear downstream task test that self-supervision papers use for comparison:
  * Input labels as a dictionary mapping filepaths to labels
  * At the end of each epoch (or whenever `trainer.evaluate()` is called), `patchwork` will do a train/test split on your labels (the split is deterministic to make sure it's consistent across runs), compute features using flattened outputs of the FCN, and train a linear support vector classifier on the features. The test accuracy and confusion matrix are recorded for TensorBoard.
  * The linear classification accuracy and model hyperparameters are saved for the TensorBoard HPARAMS interface, so you can visualize the effect of your hyperparameter choices. 
  * The `trainer.save_projections()` will record embeddings, as well as image sprites and metadata for the TensorBoard projector. I've sometimes found this to be a helpful diagnostic tool when I'm *really* stuck.
  * I can't stress enough the importance of doing actual tests on a downstream task, even if it's a small number of labels. It's *very* easy for some of these methods to learn shortcuts; they may also learn semantics that are real that just aren't useful for your problem.

![alt text](hparams.png)

Click [here](ucmerced.md) for an example comparing these methods on the UCMerced Land Use dataset.


## Context Encoder

The `patchwork.feature` module contains an implementation of the algorithm in Deepak Pathak *et al*'s [Context Encoders: Feature Learning by Inpainting](https://arxiv.org/abs/1604.07379). 

The trainer can input a list of test files- a small out-of-sample dataset is handy for visualizing how well the inpainter can reconstruct images (as opposed to just memorizing your data).

### Differences between the paper and `patchwork`

My implementation is missing the amplified loss function in the border region between masked and unmasked areas.

### Example code

```{python}
import tensorflow as tf
import patchwork as pw
import json

# load paths to train and test files
trainfiles = [x.strip() for x in open("mytrainfiles.txt").readlines()]
testfiles = [x.strip() for x in open("mytestfiles.txt").readlines()]
labeldict = json.load(open("dictionary_mapping_some_images_to_labels.json"))

# initialize trainer and train
trainer = pw.feature.ContextEncoderTrainer(
    "logs/",
    trainfiles,
    testdata=testfiles,
    num_parallel_calls=6,
    augment=aug_params,
    lr=1e-3,
    batch_size=32,
    imshape=(255,255),
    downstream_labels=labeldict
)
```

Tensorboard logs will be stored for the loss function on `testfiles` as well as visualization on inpainting:

![alt text](inpainting.png)



### Some notes on using Context Encoders

* CE can overfit easily on small datasets (and increasing the noise added by augmentation can help considerably). Reserving a small out-of-sample dataset to measure reconstruction loss on can help you identify when this is happening.



## DeepCluster

`patchwork` contains a TensorFlow implementation of the algorithm in Mathilde Caron *et al*'s [Deep Clustering for Unsupervised Learning of Visual Features](https://arxiv.org/abs/1807.05520).

### Differences between the paper and `patchwork`

I tried to keep the parameters as similar to the paper as possible. One additional keyword argument is `mult` (integer, default 1). Caron *et al* refit the k-means algorithm once per epoch on Imagenet (with `batch_size=256` this works out to roughly 5000 steps between refitting the pseudolabels). If you're working on a significantly smaller dataset you may need to run through it multiple times before refitting; set `mult=N` to refit once every `N` epochs.

### Example code

```{python}
import tensorflow as tf
import patchwork as pw

# load paths to train files
trainfiles = [x.strip() for x in open("mytrainfiles.txt").readlines()]
labeldict = json.load(open("dictionary_mapping_some_images_to_labels.json"))

# initialize a feature extractor
fcn = patchwork.feature.BNAlexNetFCN()

# train
dctrainer = pw.feature.DeepClusterTrainer(
        "logs/",
        fcn=fcn,
        trainingdata=trainfiles,
        augment=True, # or pass a dict of aug parameters here
        pca_dim=256
        k=250,
        dense=[1024, 1024],
        lr=1e-3,
        lr_decay=100000,
        batch_size=64,
        num_parallel_calls=6,
        downstream_labels=labeldict
    )

dctrainer.fit(10)
```

### Some notes on using DeepCluster

* I notice much faster training when I transfer learn from a network pretrained on ImageNet than with random weights (within 10 epochs rather than hundreds on UCMerced, for example)
* Transfer learning from a network pretrained with DeepCluster, using different parameters, can give weird results.
        

## SimCLR

`patchwork` contains a TensorFlow implementation of the algorithm in Chen *et al*'s [A Simple Framework for Contrastive Learning of Visual Representations](https://arxiv.org/abs/2002.05709). 

### Differences between the paper and `patchwork`

I haven't implemented the LARS optimizer used in the paper. Using the `opt_type` kwarg you can choose Adam or momentum optimizers.

### Example code
   
```{python}
import tensorflow as tf
import patchwork as pw

# load paths to train files
trainfiles = [x.strip() for x in open("mytrainfiles.txt").readlines()]
labeldict = json.load(open("dictionary_mapping_some_images_to_labels.json"))

# initialize a feature extractor
fcn = tf.keras.applications.ResNet50V2(weights=None, include_top=False)

# choose augmentation parameters
aug_params = {'gaussian_blur': 0.25,
             'drop_color': 0.2,
             'gaussian_noise': 0.2,
             'sobel_prob': 0.15,
             'brightness_delta': 0.2,
             'contrast_delta': 0.4,
             'saturation_delta': 0.1,
             'hue_delta': 0.1,
             'flip_left_right': True,
             'flip_up_down': True,
             'rot90': True,
             'zoom_scale': 0.4,
             'mask': 0.25}
# generally a good idea to visualize what your augmentation is doing
pw.viz.augplot(trainfiles, aug_params)   


# train
trainer = pw.feature.SimCLRTrainer(
    logdir,
    trainfiles,
    testdata=testfiles,
    fcn=fcn,
    num_parallel_calls=6,
    augment=aug_params,
    lr=1e-4,
    lr_decay=0,
    temperature=0.1,
    num_hidden=128,
    output_dim=32,
    batch_size=32,
    imshape=(256,256),
    downstream_labels=downstream_labels
)

trainer.fit(10)
```

### Some notes on using SimCLR


Note that SimCLR is much more critically dependent on image augmentation for its learning than Context Encoders or DeepCluster, so it's worth the time to experiment to find a good set of augmentation parameters. My experience so far is that SimCLR benefits from more aggressive augmentation than you'd use for supervised learning.
     
    
## Distributed Training

I've started adding multi-GPU support; this part of TensorFlow is still evolving so the details may change. The changes to your workflow are pretty minimal:

1. Initialize a "strategy" object from `tf.distribute`
* Define or load your base feature extractor within `strategy.scope()`
* Pass the strategy object to the Trainer object

Everything else should work the same. The batch size specified is the **global** batch size. I've only tested with the `MirroredStrategy()` so far.

This functionality is currently only implemented for `SimCLRTrainer` and
`MultiTaskTrainer`.

```
# same basic setup as before, but....

strat = tf.distribute.MirroredStrategy()

# check this- it should return the number of GPUs if it's working correctly
assert strat.num_replicas_in_sync == number_of_gpus_i_was_expecting

with strat.scope():
    # initialize a new model
    fcn = tf.keras.applications.ResNet50V2(weights=None, include_top=False)
    # and/or transfer learn from your last one
    fcn.load_weights("my_previous_model.h5")

trainer = pw.feature.SimCLRTrainer(
    logdir,
    trainfiles,
    testdata=testfiles,
    fcn=fcn,
    num_parallel_calls=6,
    augment=aug_params,
    lr=1e-4,
    lr_decay=0,
    temperature=0.1,
    num_hidden=128,
    output_dim=64,
    batch_size=32,
    imshape=(256,256),
    downstream_labels=downstream_labels,
    strategy=strat
)  
trainer.fit(10) 
```
    
              
## Momentum Contrast

If the hardware you're using doesn't have enough memory for the giant batch sizes that SimCLR prefers, He *et al*'s [Momentum Contrast for Unsupervised Visual Representation Learning](https://arxiv.org/abs/1911.05722) may give better results, as it decouples the number of contrastive comparisons from the batch size.

### Differences between the paper and `patchwork`

The MoCo paper has some discussion on how batch normalization can cause their algorithm to learn a shortcut; their solution is to shuffle embeddings across GPUs for comparison (and then shuffling back after).

The `patchwork` implementation attempts to accomplish the same effect while requiring only one GPU- each batch is divided in half, the halves passed through the network, and then projected representations are reassembled before computing the contrastive loss. The division used for the key network is different from the division used for the query network, so that each direct comparison between augmented images is sampled from different batch statistics.

I have not yet implemented distributed training for MoCo.

### Example code


```{python}
import tensorflow as tf
import patchwork as pw

# load paths to train files
trainfiles = [x.strip() for x in open("mytrainfiles.txt").readlines()]
labeldict = json.load(open("dictionary_mapping_some_images_to_labels.json"))

# initialize a feature extractor
fcn = tf.keras.applications.ResNet50V2(weights=None, include_top=False)

# choose augmentation parameters
aug_params = {'jitter':1,
             'gaussian_blur': 0.25,
             'flip_left_right': True,
             'zoom_scale': 0.1,}
# generally a good idea to visualize what your augmentation is doing
pw.viz.augplot(trainfiles, aug_params)   


# train
trainer = pw.feature..MomentumContrastTrainer(
        log_dir,
        trainfiles,
        fcn=fcn,
        num_parallel_calls=6,
        augment=aug_params,
        lr=3e-2,
        weight_decay=1e-4,
        decay_type="staircase",
        lr_decay=10000,
        alpha=0.999,
        tau=0.07,
        batches_in_buffer=50,
        num_hidden=512,
        output_dim=64, 
        batch_size=128,
        imshape=(128,128),
        downstream_labels=labeldict
    )

trainer.fit(10)
```

# Deprecated
        
These feature extractors are coded up in `patchwork`, but either (1) seem to have been superceded by other inventions or (2) I personally found them finicky to work with. They are unlikely to be developed further.

## Invariant Information Clustering

[paper here](https://arxiv.org/abs/1807.06653)
    