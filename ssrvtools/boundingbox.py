# -*- coding: utf-8 -*-
import numpy as np

def bbox(img):
    """Compute bounding box for given ndarray.

    Args:
        img (ndarray): Input image.
    Returns:
        (np.array, np.array): Bounding box of input image. (bbox_min, bbox_max)
    """
    if np.count_nonzero(img) == 0:
        raise ValueError('Input image is empty.')
    dim = img.ndim
    bb = np.array([np.where(np.any(img, axis=tuple([i for i in range(dim) if i != d])))[0][[0,-1]] for d in range(dim)])
    return bb[:,0],bb[:,1]

def crop(image, bbox, margin=0):
    """Crop image using given bounding box.

    Args:
        img (ndarray): Input image.
        bbox (np.array, np.array): Input bounding box.
        margin (int): The size of margin.
    Returns:
        np.ndarray: Cropped image.
    """
    bmin = bbox[0]
    bmax = bbox[1]
    bmin = np.maximum(0,bmin-margin)
    bmax = np.minimum(np.array(image.shape),bmax+(margin+1))
    dim = len(bmin)
    slice_str = ','.join(['bmin[{0}]:bmax[{0}]'.format(i) for i in range(dim)])
    return eval('image[{0}].copy()'.format(slice_str))

def trim(image, margin=0):
    """Trim image.

    This function is equivalent to ``crop(image,bbox(image),margin)``

    Args:
        img (ndarray): Input image.
        margin (int): The size of margin.
    Returns:
        np.ndarray: Trimmed image.
    """
    bb = bbox(image)
    return crop(image, bb, margin)

def uncrop(image,original_shape,bbox,margin=0,constant_values=0):
    '''Revert cropping

    Args:
        image (ndarray): Input cropped image.
        bbox (np.array, np.array): Bounding box used for cropping.
        margin (int): Margin used for cropping.
        constant_values (int or array_like): Passed to np.pad
    Returns:
        np.ndarray: Uncropped image.
    '''
    before = np.maximum(bbox[0]-margin,0)
    after = np.maximum(np.array(original_shape)-bbox[1]-margin-1,0)
    pad_width = np.array((before,after)).T
    return np.pad(image,pad_width,'constant',constant_values=constant_values)
