# Using `patchwork` loaders with `keras`

The `pw.loaders.dataset()` function returns two things: a TensorFlow dataset and
the number of steps per epoch. It looks like the steps-per-epoch input may not
be required any more in TF2.1 so I might be able to get rid of that part soon.

```
import tensorflow as tf
import patchwork as pw


# ---------- GET YOUR DATA TOGETHER ----------

# inputs: paths to image files, labels
trainfiles = ["/imdir/img01.png", "imdir/img02.jpg", ...]
labels = np.array([0,2,...])


# ---------- BUILD A KERAS MODEL ----------
model = tf.keras.Sequential([
                # layers here
])
model.compile("adam",
                loss=tf.keras.losses.sparse_categorical_crossentropy,
                metrics=["accuracy"])

# ---------- CHOOSE AUGUMENTATION STRATEGY ----------

# choose how you want data augmented, or use aug_params=True for
# defaults or aug_params=False to disable              
aug_params = {'gaussian_blur': 0.25,
             'drop_color': 0.2,
             'gaussian_noise': 0.2,
             'brightness_delta': 0.2,
             'contrast_delta': 0.4,
             'saturation_delta': 0.1,
             'flip_left_right': True,
             'rot90': True,
             'zoom_scale': 0.2}
# check to see what augmentation is doing
pw.viz.augplot(trainfiles, aug_params)

# ---------- BUILD A DATASET ----------
dataset, num_steps = pw.loaders.dataset(trainfiles, ys=labels,
                                        imshape=(256,256), num_channels=3,
                                        batch_size=32,
                                        augment=aug_params,
                                        num_parallel_calls=4,
                                        shuffle=True)

# ---------- GET TRAININ ----------
model.fit(dataset, steps_per_epoch=num_steps)

```