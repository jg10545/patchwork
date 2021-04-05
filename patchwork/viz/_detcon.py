# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from patchwork.feature._detcon_utils import _get_segments, _get_grid_segments
from patchwork.feature._detcon_utils import _segment_aug, _filter_out_bad_segments
from patchwork.feature._detcon_utils import SEG_AUG_FUNCTIONS
from patchwork.loaders import _image_file_dataset
from patchwork._augment import augment_function



def detcon_input_pipeline(imfiles, augment, mean_scale=1000, num_samples=16, 
                          outputsize=None, imshape=(256,256), **kwargs):
    """
    """
    # build a dataset to load images one at a time
    ds = _image_file_dataset(imfiles, shuffle=False, imshape=imshape,
                             **kwargs).prefetch(1)
    # place to store some examples
    img1s = []
    img2s = []
    seg1s = []
    seg2s = []

    N = len(imfiles)
    not_enough_segs_count = 0
    skipped_for_bad_segmentation = 0

    progressbar = tqdm(total=N)
    # for each image
    for x in ds:
        # get the segments
        if mean_scale > 0:
            seg, enough_segs = _get_segments(x, mean_scale=mean_scale,
                                                     num_samples=num_samples,
                                                     
                                             return_enough_segments=True)
            # count how many times we had to sample with replacement
            if not enough_segs:
                not_enough_segs_count += 1
        else:
            seg = _get_grid_segments(imshape, num_samples)
      
        # now augment image and segmentation together, twice
        img1, seg1 = _segment_aug(x, seg, augment, outputsize=outputsize)
        img2, seg2 = _segment_aug(x, seg, augment, outputsize=outputsize)
    
        # check to see if any segments were pushed out of the image by augmentation
        segmentation_ok = _filter_out_bad_segments(img1, seg1, img2, seg2)
        if not segmentation_ok:
            skipped_for_bad_segmentation += 1
        
        else:
            # finally, augment images separately
            aug2 = {k:augment[k] for k in augment if k not in SEG_AUG_FUNCTIONS}
            _aug = augment_function(imshape, aug2)
            img1 = _aug(img1).numpy()
            img2 = _aug(img2).numpy()
            seg1 = seg1.numpy()
            seg2 = seg2.numpy()
            if len(img1s) < 16:
                img1s.append(img1)
                img2s.append(img2)
                seg1s.append(seg1)
                seg2s.append(seg2)
        progressbar.update()
    
    img1 = np.stack(img1s, 0)
    img2 = np.stack(img2s, 0)
    seg1 = np.stack(seg1s, 0)
    seg2 = np.stack(seg2s, 0)
    progressbar.close()
    print(f"Had to sample with replacement for {not_enough_segs_count} of {N} images")
    print(f"Had to skip {skipped_for_bad_segmentation} of {N} images due to augmentation")

    for j in range(8):
        plt.subplot(4,4,2*j+1)
        plt.imshow(img1[j])
        plt.imshow(seg1[j].argmax(-1), alpha=0.4, extent=[0,imshape[0],imshape[1],0], cmap="tab20")
        plt.axis(False);

        plt.subplot(4,4,2*j+2)
        plt.imshow(img2[j])
        plt.imshow(seg2[j].argmax(-1), alpha=0.4, extent=[0,imshape[0],imshape[1],0], cmap="tab20")
        plt.axis(False);