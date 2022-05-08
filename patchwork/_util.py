import numpy as np
from PIL import Image
from scipy.spatial.distance import cdist
import tensorflow as tf

from patchwork._optimizers import CosineDecayWarmup, LARSOptimizer

def shannon_entropy(x, how="max"):
    """
    Shannon entropy of a 2D array
    
    :how: str; aggregate across columns by "max" or "sum"
    """
    xprime = np.maximum(np.minimum(x, 1-1e-5), 1e-5)
    elemwise_ent = -1*(xprime*np.log2(xprime)+(1-xprime)*np.log2(1-xprime))
    if how == "sum":
        return np.sum(elemwise_ent, axis=1)
    elif how == "max":
        return np.max(elemwise_ent, axis=1)
    else:
        assert False, "argument %s not recognized"%how



def tiff_to_array(f, swapaxes=True, norm=255, num_channels=-1):
    """
    Utility function to load a TIFF or GeoTIFF file to a numpy array,
    wrapping gdal.
    
    :f: string; path to file
    :swapaxes: expect that the image will be read to a (channels, height, width)
        array that needs to be reordered to (height, width, channels)
    :norm: if not None, cast raster to float and divide by this
    :num_channels: if -1, loadall channels; otherwise load the specified number 
        (e.g. 1 or 3 for display)
    """
    try:
        from osgeo import gdal
    except:
        print("couldn't load gdal")
    infile = gdal.Open(f)
    im_arr = infile.ReadAsArray()
    if swapaxes:
        im_arr = np.swapaxes(im_arr, 0, -1)
    if norm is not None:
        im_arr = im_arr.astype(np.float32)/norm
        
    if num_channels > 0:
        if num_channels == 1:
            im_arr = im_arr[:,:,0]
        else:
            im_arr = im_arr[:,:,:num_channels]
        
    return im_arr


def _load_img(f, norm=255, num_channels=3, resize=None):
    """
    Generic image-file-to-numpy-array loader. Uses filename
    to choose whether to use PIL or GDAL.
    
    :f: string; path to file
    :norm: value to normalize images by
    :num_channels: number of input channels (for GDAL only)
    :resize: pass a tuple to resize to
    """
    if type(f) is not str:
        f = f.numpy().decode("utf-8") 
    if ".tif" in f:
        img_arr = tiff_to_array(f, swapaxes=True, 
                             norm=norm, num_channels=num_channels)
    else:
        img = Image.open(f)
        img_arr = np.array(img).astype(np.float32)/norm
        
    if resize is not None:
        img_arr = np.array(Image.fromarray(
                (255*img_arr).astype(np.uint8)).resize((resize[1], resize[0]))
                    ).astype(np.float32)/norm
    # single-channel images will be returned by PIL as a rank-2 tensor.
    # we need a rank-3 tensor for convolutions, and may want to repeat
    # across 3 channels to work with standard pretrained convnets.
    if len(img_arr.shape) == 2:
        img_arr = np.stack(num_channels*[img_arr], -1)
    return img_arr.astype(np.float32)  # if norm is large, python recasts the array as float64



def compute_l2_loss(*models):
    """
    Compute squared L2-norm for all non-batchnorm trainable
    variables in one or more Keras models
    """
    loss = 0
    for m in models:
        for v in m.trainable_variables:
            if "kernel" in v.name:
                loss += tf.nn.l2_loss(v)
                
    return loss


def _compute_alignment_and_uniformity(dataset, model, alpha=2, t=2):
    """
    Compute measures from "Understanding Contrastive Representation Learning
    Through Alignment and Uniformity on the Hypersphere" by Wang and Isola.
    
    :dataset: tensorflow dataset returning augmented pairs of images
    :model: keras model mapping images to semantic vectors (NOT the fully 
              convolutional network!)
    :alpha: parameter for alignment
    :t: parameter for uniformity
    
    Returns alignment, uniformity
    """
    pool = tf.keras.layers.GlobalAvgPool2D()
    batch_alignment = []
    vectors = []
    for x1,x2 in dataset:
        # compute normalized embedding vectors for each
        z1 = tf.nn.l2_normalize(pool(model(x1)), axis=1)
        z2 = tf.nn.l2_normalize(pool(model(x2)), axis=1)
        batch_alignment.append(tf.reduce_sum((z1-z2)**alpha, axis=1).numpy())
        vectors.append(z1.numpy())

    alignment = np.concatenate(batch_alignment, axis=0).mean()

    vectors = np.concatenate(vectors, axis=0)
    uniformity = np.log(np.mean(np.exp(-1*t*cdist(vectors, vectors, metric="euclidean"))))
    return alignment, uniformity



def build_optimizer(lr, lr_decay=0, opt_type="adam", decay_type="exponential",
                    weight_decay=None):
    """
    Macro to reduce some duplicative code for building optimizers
    for trainers
    
    :lr: float; initial learning rate
    :lr_decay: int; learning rate decay steps (0 to disable)
    :opt_type: str; type of optimizer. "momentum", "adam", or "lars"
    :decay_type: str; type of decay. "exponential", "staircase", "cosine", or "warmupcosine"
    :weight_decay: float; weight decay parameter. ONLY APPLIES TO LARS
    """
    if lr_decay > 0:
        if decay_type == "exponential":
            lr = tf.keras.optimizers.schedules.ExponentialDecay(lr, 
                                        decay_steps=lr_decay, decay_rate=0.5,
                                        staircase=False)
        elif decay_type == "staircase":
            lr = tf.keras.optimizers.schedules.ExponentialDecay(lr, 
                                        decay_steps=lr_decay, decay_rate=0.5,
                                        staircase=True)
        elif decay_type == "cosine":
            lr = tf.keras.experimental.CosineDecayRestarts(lr, lr_decay,
                                                           t_mul=2., m_mul=1.,
                                                           alpha=0.)
        elif decay_type == "warmupcosine":
            lr = CosineDecayWarmup(lr, lr_decay)
        else:
            assert False, "don't recognize this decay type"
    if opt_type == "adam":
        opt = tf.keras.optimizers.Adam(lr)
    elif opt_type == "momentum":
        opt = tf.keras.optimizers.SGD(lr, momentum=0.9)
    elif opt_type == "lars":
        if weight_decay is None:
            kwargs = {}
        else:
            kwargs = {"weight_decay":weight_decay}
        opt = LARSOptimizer(lr, **kwargs)
    else:
        assert False, "dont know what to do with {}".format(opt_type)
        
    # if we're using mixed precision, do automated loss scaling
    if tf.keras.mixed_precision.global_policy().name == 'mixed_float16':
        opt = tf.keras.mixed_precision.LossScaleOptimizer(opt)
        
    return opt
