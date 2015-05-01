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
import cPickle

class mvpa_mat(object):
    '''MVPA matrix of trials by features'''
    def __init__(self, data, subject_name, mask_name, mask_index, class_labels):
        self.data = data
        self.subject_name = subject_name
        self.mask_name = mask_name
        self.mask_index = mask_index        
        self.n_features = self.data.shape[1]
        self.n_trials = self.data.shape[0]
        self.class_labels = class_labels        
        self.class_names = list(set(self.class_labels))

    def normalize_mvpa(method):
        ''' 
        Normalizes mvpa matrix in a univariate (t-stat) or 
        multivariate approach
        '''
        
        if method == 'univariate':
            
            
        if method == 'multivariate':
            pass
        
        

def extract_class_vector(subject_directory):
    """ Extracts class of each trial and returns a vector of class labels."""
    
    sub_name = os.path.basename(os.path.normpath(subject_directory))
    to_parse = subject_directory + '/design.con'
    
    # Read in design.con
    if os.path.isfile(to_parse):
        with open(to_parse, 'r') as con_file:
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
    
def create_subject_mats(mask, subject_stem):
    """ 
    Creates subject-specific MVPA matrices and stores them
    in individual dictionaries. 
    
    Args: 
    firstlevel_dir  = directory with individual firstlevel data
    mask            = mask to index constrasts, either 'fstat' or a specific 
                      ROI. The exact name should be given 
                      (e.g. 'graymatter.nii.gz').
    subject_stem    = project-specific subject-prefix
   
    Returns:
    Nothing, but creates a dir ('mvpa_mats') with individual pickle files.
    """
    firstlevel_dir = os.getcwd()
    subject_dirs = glob.glob(os.getcwd() + '/*' + subject_stem + '*')    
    
    mat_dir = os.getcwd() + '/mvpa_mats'
    
    if not os.path.exists(mat_dir):
        os.makedirs(mat_dir)
    
    # Load mask, create index
    mask_name = os.path.basename(mask)
    mask_vol = nib.load(mask)
    mask_index = mask_vol.get_data().ravel() > 0
    n_features = np.sum(mask_index)    
    
    for i_sub, sub_path in enumerate(subject_dirs):
        
        # Extract class vector
        class_labels = extract_class_vector(sub_path)
        
        sub_name = os.path.basename(os.path.normpath(sub_path))
        print 'Processing ' + sub_name
        
        # load in dirNames
        stat_paths = glob.glob(sub_path + '/stats_new/cope*mni*.nii.gz')
        n_stat = len(stat_paths)

        if not n_stat == len(class_labels):
            raise ValueError('The number of trials do not match the number ' \
                             'of class labels')
        elif n_stat == 0:
            raise ValueError('There are no valid MNI COPES in ' + os.getcwd())
        
        # Pre-allocate
        mvpa_data = np.zeros([n_stat,n_features])

        # Load in data
        for i, path in enumerate(stat_paths):
            data = nib.load(path).get_data()
            mvpa_data[i,:] = np.ravel(data)[mask_index]

        to_save = mvpa_mat(mvpa_data, sub_name, mask_name, mask_index, class_labels) 
        
        with open(mat_dir + '/' + sub_name + '.cPickle', 'wb') as handle:
            cPickle.dump(to_save, handle)
             
    print 'Created ' + str(len(glob.glob(mat_dir + '/*.cPickle'))) + ' MVPA matrices' 
    
