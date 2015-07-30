# -*- coding: utf-8 -*-
"""
Module to create subject-specific MVPA matrices of trials X voxels.
Contains:
1. mvpa_mat class;
2. create_subject_mats (main mvpa_mat creation);
3. Some uninteresting but necessary file parsing/transforming functions

The creat_subject_mats function does the following for each subject:
1. Loads in mni-transformed first-level COPEs
2. Indexes vectorized copes with specified mask (e.g. ROI/gray matter)
3. Normalizes COPEs by their variance (sqrt(VARCOPE)); this will be extended 
with a multivariate normalization technique in the future
4. Initializes the result as an mvpa_mat
5. Saves subject-specific mvpa_mat as .cpickle file 

Lukas Snoek, master thesis Dynamic Affect, 2015
"""

_author_ = "Lukas"

import os
import numpy as np
import nibabel as nib
import glob
import csv
import cPickle
from sklearn import preprocessing as preproc
import fnmatch

class mvpa_mat():
    '''MVPA matrix of trials by features'''
    def __init__(self, data, subject_name, mask_name, mask_index, mask_shape, class_labels):
        # Initialized attributes        
        self.data = data
        self.subject_name = subject_name
        self.mask_name = mask_name
        self.mask_index = mask_index        
        self.mask_shape = mask_shape
        self.class_labels = class_labels        
        
        # Computed attributes (based on initialized attributes)
        self.n_features = self.data.shape[1]
        self.n_trials = self.data.shape[0]
        
        class_names = []
        [class_names.append(i) for i in class_labels if not class_names.count(i)]
        self.class_names = class_names
        self.n_class = len(class_names)        
        self.n_inst = self.n_trials / self.n_class
        self.class_idx = [np.arange(self.n_inst*i,self.n_inst*i+self.n_inst) \
                          for i in range(self.n_class)]

        # Create numeric label vector
        n_class = len(class_names)
        num_labels = []        
        
        for i in xrange(1, n_class+1):
            for j in xrange(1, self.n_inst+1):
                num_labels.append(1*i)

        self.num_labels = np.asarray(num_labels)

    def normalize(self, style):
        if style == 'z':
            self.data = normalize(self.data)
        
def extract_class_vector(sub_path, remove_class):
    """ Extracts class of each trial and returns a vector of class labels."""
    
    sub_name = os.path.basename(os.path.normpath(sub_path))
    to_parse = os.path.join(subject_directory,'design.con')
    
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

        remove_idx = []
        for match in remove_class:
            to_remove = fnmatch.filter(class_labels,'*%s*' % (match))
            
            for name in to_remove:
                remove_idx.append(class_labels.index(name))
            
        class_labels = [s.split('_')[0] for s in class_labels]
    
        return(class_labels)
                    
    else:
        print('There is no design.con file for ' + sub_name)
    
def create_subject_mats(mask, subject_stem, mask_threshold,norm_method = 'nothing'):
    """ 
    Creates subject-specific MVPA matrices, initializes them as an
    mvpa_mat object and saves them as a cpickle file.
    
    Args: 
    firstlevel_dir  = directory with individual firstlevel data
    mask            = mask to index constrasts, either 'fstat' or a specific 
                      ROI. The exact name should be given 
                      (e.g. 'graymatter.nii.gz').
    subject_stem    = project-specific subject-prefix
    mask_threshold  = min. threshold of probabilistic FSL mask
   
    Returns:
    Nothing, but creates a dir ('mvpa_mats') with individual pickle files.
    
    Lukas Snoek    
    """
    
    data_dir = os.path.join(os.getcwd(), '*%s*' % (subject_stem),'*.feat') 
    
    subject_dirs = glob.glob(data_dir)
    mat_dir = os.path.join(os.getcwd(),'mvpa_mats')
    
    if not os.path.exists(mat_dir):
        os.makedirs(mat_dir)
    
    # Load mask, create index
    mask_name = os.path.basename(mask)
    mask_vol = nib.load(mask)
    mask_shape = mask_vol.get_shape()
    mask_index = mask_vol.get_data().ravel() > mask_threshold
    n_features = np.sum(mask_index)    
    
    for sub_path in subject_dirs:
        
        # Extract class vector (see definition below)
        class_labels = extract_class_vector(sub_path)
        
        sub_name = os.path.basename(os.path.dirname(sub_path))
        
        print 'Processing ' + sub_name + ' ... ',
        
        # Generate and sort paths to stat files (COPEs)
        
        stat_paths = glob.glob(os.path.join(sub_path,'reg_standard','cope*.nii.gz'))
        stat_paths = sort_stat_list(stat_paths) # see function below
        n_stat = len(stat_paths)

        if not n_stat == len(class_labels):
            raise ValueError('The number of trials do not match the number ' \
                             'of class labels')

        if n_stat == 0: 
            raise ValueError('There are no valid MNI COPES in ' + os.getcwd())
        
        # Pre-allocate
        mvpa_data = np.zeros([n_stat,n_features])

        # Load in data (COPEs)
        for i, path in enumerate(stat_paths):
            cope = nib.load(path).get_data()
            mvpa_data[i,:] = np.ravel(cope)[mask_index]

        ''' NORMALIZATION OF VOXEL PATTERNS '''
        if norm_method == 'nothing':
            pass
        
        if norm_method == 'univariate':
            varcopes = glob.glob(os.path.join(sub_path,'reg_standard','varcope*.nii.gz'))
            varcopes = sort_stat_list(varcopes)
            
            for i_trial, varcope in enumerate(varcopes):
                var = nib.load(varcope).get_data()
                var_sq = np.sqrt(var.ravel()[mask_index])
                mvpa_data[i_trial,] = mvpa_data[i_trial,] / var_sq
                
        if norm_method == 'multivariate':
            res4d = nib.load(sub_path + '/stats_new/res4d_mni.nii.gz').get_data()
            res4d.resize([np.prod(res4d.shape[0:3]), res4d.shape[3]])
            res4d = res4d[mask_index,]            
        
            #res_cov = np.cov(res4d)
        
        mvpa_data = preproc.scale(mvpa_data)
        
        # Initializing mvpa_mat object, which will be saved as a pickle file
        to_save = mvpa_mat(mvpa_data, sub_name, mask_name, mask_index, mask_shape, class_labels) 
        filename = os.path.join(mat_dir, '%s.cPickle' % (sub_name))
        
        with open(filename, 'wb') as handle:
            cPickle.dump(to_save, handle)
    
        print 'done.'
    print 'Created %i MVPA matrices' %  len(glob.glob(mat_dir + '/*.cPickle'))

def sort_stat_list(stat_list):
    """
    Sorts list with paths to statistic files (e.g. COPEs, VARCOPES),
    which are often sorted wrong (due to single and double digits).
    This function extracts the numbers from the stat files and sorts 
    the original list accordingly.
    """
    num_list = []    
    for path in stat_list:
        num = [str(s) for s in str(os.path.basename(path)) if s.isdigit()]
        num_list.append(int(''.join(num)))
    
    sorted_list = [x for y,x in sorted(zip(num_list, stat_list))]
    return(sorted_list)
    
def merge_runs():
    
    sub_paths = [os.path.abspath(path) for path in glob.glob(os.getcwd() + '/mvpa_mats/*WIPPM*cPickle*')]
    abbr = [os.path.basename(path)[6] for path in sub_paths]    
    sub_paths = [x for y,x in sorted(zip(abbr, sub_paths))]
    
    n_sub = len(sub_paths)
    
    i = 0
    for dummy in xrange(n_sub/2):
        run1 = cPickle.load(open(sub_paths[i]))
        run2 = cPickle.load(open(sub_paths[i+1]))
        
        data = np.zeros((run1.n_trials*2, run1.n_features))
        class_labels = []
        
        j = 0
        for k in xrange(run1.n_trials-1):
            data[j,:] = run1.data[k,:]
            data[j+1,:] = run2.data[k,:]
                        
            class_labels.append(run1.class_labels[k])
            class_labels.append(run2.class_labels[k])
            
            j += 2
        
        class_labels.append(run1.class_labels[k+1])
        class_labels.append(run2.class_labels[k+1])
        
        name = os.path.basename(sub_paths[i])[0:7] + '_merged'
        mask_name = run1.mask_name
        mask_index = run1.mask_index
        mask_shape = run1.mask_shape
        
        to_save = mvpa_mat(data,name,mask_name, mask_index, mask_shape, class_labels)
        
        with open(os.getcwd() + '/' + name + '.cPickle', 'wb') as handle:
            cPickle.dump(to_save, handle)
            
        i += 2
        print "Merged subject %i " % dummy