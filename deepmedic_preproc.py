#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 10:47:53 2021

@author: xgryem

The script contains functions without input/output routines.
"""

import os
import numpy as np
import nibabel as nib

def Nii2Arr(image_path):
    
    """Load nifti file from image_path; 
    return numpy array of the image and nifti image affine"""

    img_nii = nib.load(image_path)
    img_affine = img_nii.affine
    img_arr = np.squeeze(img_nii.get_fdata())

    return img_arr, img_affine
    
def ZScoreNormalization(image):
    
    """ Calculate the mean and standard deviation of a numpy array;
    return normalized numpy array"""

    image_mean = np.mean(image[image!=0])
    image_variance = np.std(image[image!=0])
    normalized_image = (image-image_mean) / image_variance
    
    return normalized_image

def GetRoiMask(image):

    """Create a binary brain mask by thresholding intensity values > 0"""

    mask = image.copy()
    mask[mask > 0] = 1

    return mask
