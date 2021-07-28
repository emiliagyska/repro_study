#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 28 11:53:34 2021

@author: xgryem
"""

import os
import shutil
import numpy as np
import nibabel as nib
import SimpleITK as sitk
from scipy.interpolate import interp1d
from skimage.morphology import dilation, cube
from nipype.interfaces.ants import N4BiasFieldCorrection
from func import Mha2Nii, Nii2Arr

def AntsN4ITK(in_path, out_path):
    
    n4 = N4BiasFieldCorrection()
    n4.inputs.save_bias = False
   
    # parameters used by S. Pereira et al.:
    n4.inputs.dimension = 3
    n4.inputs.bspline_fitting_distance = 200
    n4.inputs.shrink_factor = 2
    n4.inputs.n_iterations = [20, 20, 20, 10]
    n4.inputs.convergence_threshold = 0
    
    # I/O
    n4.inputs.input_image = in_path
    n4.inputs.output_image = out_path
    n4.run()

def IntensityNormalization(input_path, output_path):
    
    file_name = os.path.basename(input_path)
    idx = np.where(['MR_' in item for item in file_name.split('.')])[0][0]
    sequence = file_name.split('.')[idx].split('MR_')[-1]
    
    img_arr, img_aff = Nii2Arr(input_path)
    
    if sequence == 'Flair':
        
        # transformation parameters according to S. Pereira et al.
        percentiles = [2.5, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 96, 98]
        landmark_intensities = [10010., 10207., 10511., 10864., 11025., 11117., 11190., 11261., 11339., 11444., 11646., 11989., 12244.]
        lower_virtual_intensity = 10010.0
        higher_virtual_intensity = 14692.8
        higher_clipping_intensity = 29385.6
        
    elif sequence == 'T1':
        
        percentiles = [2, 2.5, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 98, 98.5]
        landmark_intensities = [5004., 5151., 5661., 6293., 7045., 7486., 7792., 8043., 8275., 8507., 8751., 9033., 9238., 9479., 9564.]
        lower_virtual_intensity = 5004.0
        higher_virtual_intensity = 11476.8
        higher_clipping_intensity = 22953.6
        
    elif sequence == 'T1c':
        
        percentiles = [1, 2.5, 5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 92.5, 95, 97.5, 99.0]
        landmark_intensities = [5004., 5519., 6030., 6703., 7126., 7414., 7796., 8066., 8295., 8509., 8724., 8952., 9256., 9384., 9596., 10116., 11144.]
        lower_virtual_intensity = 5004.0
        higher_virtual_intensity = 13372.8
        higher_clipping_intensity = 26745.6
        
    elif sequence == 'T2':

        percentiles = [2.5, 5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 95, 97, 98, 99]
        landmark_intensities = [ 10014., 10756., 11341., 11601., 11786., 12086., 12376., 12690., 13059., 13544., 14305., 15776., 17191., 18070., 18645., 19455.]
        lower_virtual_intensity = 10014.0
        higher_virtual_intensity = 23346.0
        higher_clipping_intensity = 46692.0   

    else:
        pass
        
    # extract intensities at percentiles within the breain region
    brain_intensities = img_arr[img_arr > 0]
    current_intensities = np.percentile(brain_intensities, percentiles)
    f = interp1d(current_intensities, landmark_intensities, bounds_error=False, fill_value=(lower_virtual_intensity, higher_virtual_intensity))
    normalized = f(img_arr)
    #normalized[normalized > higher_clipping_intensity] = ???
    
    norm_nii = nib.Nifti1Image(normalized, img_aff)
    nib.save(norm_nii, output_path)

""" The script below copies the dorectory containing original
    nifti files and applies the N4ITK correction to the images in the copied directory"""

input_dir_n4 = '/path/to/BRATS2015/pereira_repro/original_nii'
output_dir_n4 = '/path/to/BRATS2015/pereira_repro/n4'

shutil.copytree(input_dir_n4, output_dir_n4)

# implement N4ITK correction
for path, dirs, files in os.walk(output_dir_n4):
    print(path)
    if len(files) != 0:
        for f in files:
            if f.endswith('.nii.gz'):
                fn4 = f.split('.nii.gz')[0] + '_n4.nii.gz'
                AntsN4ITK(os.path.join(path, f), os.path.join(path, fn4))
                os.remove(os.path.join(path, f))

""" The script below copies the dorectory containing N4-corrected
    nifti files and applies the Nyul's intensity normalizatin to the images in the copied directory"""

input_dir_in = '/path/to/BRATS2015/pereira_repro/n4'
output_dir_in = '/path/to/BRATS2015/pereira_repro/n4_norm/'

shutil.copytree(input_dir_in, output_dir_in)

# implement intensity normalization acc. to Nyul et al and the histograms specified by Pereira et al.
for path, dirs, files in os.walk(output_dir_in):
    if len(files) != 0:
        for f in files:
            if ('Brain_3more' in f) or ('Brain_2more' in f) or ('Brain_1more' in f):
                os.remove(os.path.join(path,f))
            else:
                print(path)
                fin = f.replace('_n4', 'nrm')
                IntensityNormalization(os.path.join(path,f), os.path.join(path,fin))
                os.remove(os.path.join(path, f))
                
""" Patch normlization """           
# get mean and st. dev. of training patches -- the lesion distribution in the training patches is unknown.
# instead, I'll extract intensity profile from diluted lesion area (diluted to include some neighbouring/normal tissue as it would be included in patches).


norm_img_path = '/path/to/BRATS2015/pereira_repro/n4_norm'
seg_path = '/path/to/BRATS2015/pereira_repro/original_nii'

list_patients = os.listdir(norm_img_path)

t1_sum_intensities = 0
t1_sum_squared = 0

t1c_sum_intensities = 0
t1c_sum_squared = 0

t2_sum_intensities = 0
t2_sum_squared = 0

fl_sum_intensities = 0
fl_sum_squared = 0

num_values = 0

for pat in list_patients:
    print(pat)
    
    ims_dir = os.path.join(norm_img_path, pat)
    seg_dir = os.path.join(seg_path, pat)
    
    seg_f = list(filter(lambda x: 'more' in x, os.listdir(seg_dir)))[0]
    seg_arr = Nii2Arr(os.path.join(seg_dir, seg_f), ret_affine=False)
    seg_arr[seg_arr != 0] = 1
    str_el = cube(25)
    diluted_seg = dilation(seg_arr, str_el)
    
    img_files = os.listdir(ims_dir)
    
    for img_f in img_files:
        img_arr = Nii2Arr(os.path.join(ims_dir, img_f), ret_affine=False)
        img_seg = img_arr*seg_arr
        
        img_seg_vec = np.around(img_seg[img_seg > 0])
        
        
        if 'T1.' in img_f:
            t1_sum_intensities += img_seg_vec.sum()
            t1_sum_squared += (img_seg_vec**2).sum()
            num_values += img_seg_vec.size
 
        elif 'T1c.' in img_f:            
            t1c_sum_intensities += img_seg_vec.sum()
            t1c_sum_squared += (img_seg_vec**2).sum()
            
        elif 'T2' in img_f:
            t2_sum_intensities += img_seg_vec.sum()
            t2_sum_squared += (img_seg_vec**2).sum()
            
        elif 'Flair' in img_f:
            fl_sum_intensities += img_seg_vec.sum()
            fl_sum_squared += (img_seg_vec**2).sum()
            

t1_mean = t1_sum_intensities/num_values
t1c_mean = t1c_sum_intensities/num_values
t2_mean = t2_sum_intensities/num_values
fl_mean = fl_sum_intensities/num_values

t1_std = np.sqrt((t1_sum_squared/num_values)-t1_mean**2)
t1c_std = np.sqrt((t1c_sum_squared/num_values)-t1c_mean**2)
t2_std = np.sqrt((t2_sum_squared/num_values)-t2_mean**2)
fl_std = np.sqrt((fl_sum_squared/num_values)-fl_mean**2)
    
avg_vec = [t1_mean, t1c_mean, t2_mean, fl_mean]
std_vec = [t1_std, t1c_std, t2_std, fl_std]
 
np.save('avg_vec_25.npy', avg_vec)
np.save('std_vec_25.npy', std_vec)           
    

# normalize testing images with mean and stdev calculated from training patches -25

test_img_path = '/path/to/BRATS2015/pereira_repro/n4_norm_patch/'

for path, dirs, files in os.walk(test_img_path):
    
    sub_id = path.split('/')[-1]
    
    for f in files:
        img, aff = Nii2Arr(os.path.join(path, f))
        
        if 'T1.' in f:
            fname = sub_id + '_t1.nii.gz'
            img_norm = (img - t1_mean)/t1_std
            norm_nii = nib.Nifti1Image(img_norm, aff)
            nib.save(norm_nii, os.path.join(path, fname))
            os.remove(os.path.join(path, f))
            
        elif 'T1c.' in f:
            fname = sub_id + '_t1c.nii.gz'
            img_norm = (img - t1c_mean)/t1c_std
            norm_nii = nib.Nifti1Image(img_norm, aff)
            nib.save(norm_nii, os.path.join(path, fname))
            os.remove(os.path.join(path, f))            
            
        elif 'T2.' in f:
            fname = sub_id + '_t2.nii.gz'
            img_norm = (img - t2_mean)/t2_std
            norm_nii = nib.Nifti1Image(img_norm, aff)
            nib.save(norm_nii, os.path.join(path, fname))
            os.remove(os.path.join(path, f))            
            
            
        if 'Flair.' in f:
            fname = sub_id + '_flair.nii.gz'
            img_norm = (img - fl_mean)/fl_std
            norm_nii = nib.Nifti1Image(img_norm, aff)
            nib.save(norm_nii, os.path.join(path, fname))
            os.remove(os.path.join(path, f))    
   



















































