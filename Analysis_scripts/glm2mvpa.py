# -*- coding: utf-8 -*-
"""
Module to create subject-specific MVPA matrices of voxels X trials.
Loops over subjects and load single trial GLM data and stores them 
in a dictionary with 'data' (ndarray) and 'trials_class' (list).

Lukas Snoek, master thesis Dynamic Affect, 2015
"""

import os
import numpy as np
import nibabel as nib
import glob
import matplotlib.pyplot as plt
import csv
import pickle
import FSLutilities

def extract_class_vector(subject_directory):
    """ Extracts class of each trial and returns a vector of class labels."""
    
    os.chdir(subject_directory)
    sub_name = os.path.basename(os.path.normpath(subject_directory))
    
    # Read in design.con
    if os.path.isfile('design.con'):
        with open('design.con', 'r') as con_file:
            con_read = csv.reader(con_file, delimiter='\t')
            class_labels = []            
            
            # Extract class of trials until /NumWaves
            for line in con_read:     
                if line[0] == '/NumWaves':
                    break
                else:
                    class_labels.append(line[1])
        
        class_labels = [s.split('_')[0] for s in class_labels]
        return(class_labels)
                    
    else:
        print('There is no design.con file for ' + sub_name)
    
    os.chdir(os.pardir)    
        
def create_subject_mats(firstlevel_dir, mask, subject_stem):
    """ 
    Creates subject-specific MVPA matrices and stores them
    in individual dictionaries. 
    
    Args: 
    firstlevel_dir  = directory with individual firstlevel data
    mask            = mask to index constrasts, either 'fstat' or a specific 
                      ROI. The exact name should be given 
                      (e.g. 'graymatter.nii.gz').
            
    Returns:
    Nothing, but creates a dir ('mvpa_mats') with individual pickle files.
    """

    os.chdir(firstlevel_dir)
    subject_dirs = glob.glob(os.getcwd() + '/*' + subject_stem + '*')    
    
    os.chdir(os.pardir)
    mat_dir = os.getcwd() + '/mvpa_mats'
    
    if not os.path.exists(mat_dir):
        os.makedirs(mat_dir)
    
    # Load mask, create index
    mask_vol = nib.load(mask)
    mask_index = mask_vol.get_data().ravel() > 0
    n_features = np.sum(mask_index)    
    
    for i_sub, sub_path in enumerate(subject_dirs):
        
        # Extract class vector
        class_labels = extract_class_vector(sub_path)
        
        sub_name = os.path.basename(os.path.normpath(sub_path))
        os.chdir(sub_path + '/reg_standard')        
        
        print 'Processing ' + sub_name
        
        # load in dirNames
        tstat_paths = glob.glob('*tstat*.nii.gz')
        n_tstat = len(tstat_paths)

        if not n_tstat == len(class_labels):
            raise ValueError('The number of trials do not match the number ' \
                             'of class labels')
        # Pre-allocate
        mvpa_mat = np.zeros([n_tstat,n_features])

        # Load in data
        for i, path in enumerate(tstat_paths):
            data = nib.load(path).get_data()
            mvpa_mat[i,:] = np.ravel(data)[mask_index]

        # Extract class_vector
        os.chdir(mat_dir)
        
        sub_dict = {'name': sub_name,
                    'data': mvpa_mat,
                    'class_labels': class_labels}
                    
        with open(sub_name + '.pickle', 'wb') as handle:
            pickle.dump(sub_dict, handle)
        
        print 'Saving MVPA data from ' + sub_name
        
    print 'Created ' + str(len(glob.glob('*.pickle'))) + ' MVPA matrices' 
    os.chdir(firstlevel_dir)
    
    mask_toWrite = mask_index.reshape(mask_vol.shape).astype(float)
    mask_outVol = nib.Nifti1Image(mask_toWrite, np.eye(4))    
    nib.save(mask_outVol, 'mask_outVol')
    
    slice_0 = mask_toWrite[26, :, :]
    slice_1 = mask_toWrite[:, 30, :]
    slice_2 = mask_toWrite[:, :, 16]
    show_slices([slice_0, slice_1, slice_2])
    plt.suptitle("Center slices for mask volume")  
   
def show_slices(slices):
    """ Function to display row of image slices """
    fig, axes = plt.subplots(1, len(slices))
    
    for i, slice in enumerate(slices):
        axes[i].imshow(slice.T, cmap="Reds", origin="lower")
        axes[i].set_xlim([0, 110])
        axes[i].set_ylim([-10, 100])
        axes[i].set_xticks([]) 
        axes[i].set_yticks([]) 
        axes[i].axis('off')