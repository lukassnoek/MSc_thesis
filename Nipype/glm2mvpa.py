# -*- coding: utf-8 -*-
"""
Module to create subject-specific MVPA matrices of voxels X trials.
Loops over subjects and load single trial GLM data and stores them 
in a dictionary with 'data' (ndarray) and 'trials_class'.
"""

import os
import numpy as np
import nibabel as nib
import glob
import matplotlib.pyplot as plt
import csv
import pickle

def extract_class_vector(subject_directory):
    """ Extracts class of each trial and returns a class_vector."""
    
    os.chdir(subject_directory)
    sub_name = os.path.basename(os.path.normpath(subject_directory))
    
    # Read in design.con
    if os.path.isfile('design.con'):
        with open('design.con', 'r') as con_file:
            con_read = csv.reader(con_file, delimiter='\t')
            class_vector = []            
            
            # Extract class of trials until /NumWaves
            for line in con_read:     
                if line[0] == '/NumWaves':
                    break
                else:
                    class_vector.append(line[1])
                    
        return(class_vector)
                    
    else:
        print('There is no design.con file for ' + sub_name)
        
def create_subject_mats(firstlevel_dir, mask):
    """ Creates subject-specific MVPA matrices."""

    os.chdir(firstlevel_dir)
    subject_dirs = glob.glob(os.getcwd() + '/*hww*')    
    
    os.chdir(os.pardir)
    mat_dir = os.getcwd() + '/mvpa_mats'
    
    if not os.path.exists(mat_dir):
        os.makedirs(mat_dir)

    for i_sub, sub_path in enumerate(subject_dirs):
        
        # Extract class vector
        trials_class = extract_class_vector(sub_path)
        
        sub_name = os.path.basename(os.path.normpath(sub_path))
        os.chdir(sub_path + '/reg_standard')        
        
        # Load mask, create index
        img = nib.load('zfstat1.nii.gz')
        mask = img.get_data()
        mask_index = mask > 2.3
        n_features = np.sum(mask_index)

        # load in dirNames
        tstat_paths = glob.glob('*tstat*.nii.gz')
        n_tstat = len(tstat_paths)

        # Pre-allocate
        mvpa_mat = np.zeros([n_features,n_tstat])

        # Load in data
        for i, path in enumerate(tstat_paths):
            data = nib.load(path).get_data()
            mvpa_mat[:,i] = np.ravel(data[mask_index])

        # Extract class_vector
        os.chdir(mat_dir)
        
        sub_dict = {'name': sub_name,
                    'data': mvpa_mat,
                    'trials_class': trials_class}
                    
        np.save(sub_name + '_mat', sub_dict)
    
    print 'Created ' + str(len(glob.glob('*.npy'))) + ' MVPA matrices' 
    os.chdir(firstlevel_dir)
    
#def show_slices(slices):
#""" Function to display row of image slices """
#    fig, axes = plt.subplots(1, len(slices))
#    for i, slice in enumerate(slices):
#        axes[i].imshow(slice.T, cmap="gray", origin="lower")

#slice_0 = data[26, :, :]
#slice_1 = data[:, 30, :]
#slice_2 = data[:, :, 16]
#show_slices([slice_0, slice_1, slice_2])
#plt.suptitle("Center slices for EPI image")  