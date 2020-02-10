
# Input and augmentation parameters

Augmentation is a critical part of effectively using self-supervision for imagery- it's worth the time to experiment with augmentation parameters to find a set that work well for your problem. Once you've got them, you can re-use them in all the `patchwork` tools.

## Input parameters

* `imshape` (H,W) tuple defining an image shape. All images will be resampled to this shape.
* `num_channels` integer; the number of channels per image. If an image has more channels than this (for example, and RGBA image when `num_channels=3`) it will be truncated.
* `norm` value to divide image data by to scale it to the unit interval. This will usually be 255 but may be different for GeoTIFFs, for example.
* `batch_size` integer; batch size for training
* `num_parallel_calls` integer; number of parallel threads to use for loading and augmenting (generally set to number of CPUs)
* `sobel` Boolean; if `True` then each image is averaged across its channels and then Sobel filtered. The 2-channel output of the Sobel filter is padded with a third channel of zeros so that you can use this with standard 3-channel-input convnets.
* `single_channel` Boolean; let `patchwork` know that you expect single-channel input images. If `num_channels > 1` the image will be repeated across channels (again, for example, for using single-channel images with 3-channel-input convnets)

## Augmentation parameters

You can pass `False` to the `augment` parameter to disable augmentation, `True` to use defaults, or a dictionary containing any of the following (with the rest disabled):


* `max_brightness_delta` (default 0.2): randomly adjust brightness within this range
* `contrast_min` (default 0.4) and `contrast_max` (default 1.4): randomly adjust contrast within this range
* `max_hue_delta` (default 0.1): randomly adjust hue within this range
* `max_saturation_delta` (default 0.5): randomly adjust saturation within this range
* `left_right_flip` (default True): if True, flip images left-to-right 50% of the time
* `up_down_flip` (default True): if True, flip images top-to-bottom 50% of the time
* `rot90` (default True): if True, rotate images 0, 90, 180, or 270 degrees with equal probability
* `zoom_scale` (default 0.3): add a random pad to each side of the image, then randomly crop from each side- this parameter sets the scale for both.
* `select_prob` (default 0.5) flip a weighted coin with this probability for each of the augmentation steps to decide whether to apply it.

Use `patchwork.viz.augplot()` to experiment with augmentation:

```{python}
aug = {"max_brightness_delta":0.3, "contrast_min":0.4, "contrast_max":1.4,
        "max_hue_delta":0.2, "max_saturation_delta":0.5,"left_right_flip":True, 
        "up_down_flip":True, "rot90":True, "zoom_scale":0.2, "select_prob":0.5}
# imfiles is a list of paths to image files
pw.viz.augplot(imfiles, aug)
```
![](augment.png)