# -*- coding: utf-8 -*-
"""
Main classification module

Lukas Snoek
"""

import numpy as np
import itertools

def draw_random_subsample(mvpa_data, n_train):
    ''' Draws random subsample of trials for each class '''
    train_ind = np.zeros(len(mvpa_data.num_labels), dtype=np.int)
    
    j = 0
    for c in mvpa_data.class_names:
        ind = np.random.choice(np.arange(j+1, j+mvpa_data.n_inst+1), n_train, replace=False)            
        train_ind[ind] = 1
        j = j + mvpa_data.n_inst
    
    return train_ind.astype(bool)
    
def select_voxels(mvpa, train_ind, threshold):
    ''' 
    Feature selection. Runs on single subject data
    '''
    
    # Setting useful parameters & pre-allocation
    n_train = np.sum(train_ind) / mvpa.n_class
    mvpa.data = mvpa.data[train_ind,] # should be moved to main_classify() later
    av_patterns = np.zeros((mvpa.n_class, mvpa.n_features))
    
    # Calculate mean patterns
    j = 0    
    for c in xrange(mvpa.n_class):         
        av_patterns[c,] = np.mean(mvpa.data[range(j,j + n_train),], axis = 0)
        j = j + n_train
    
    # Create difference vectors, z-score standardization, absolute
    # To implement: min-max scaling?
    comb = list(itertools.combinations(range(1,mvpa.n_class+1),2))
    diff_patterns = np.zeros((len(comb), mvpa.n_features))
        
    for i, cb in enumerate(comb):        
        x = np.subtract(av_patterns[cb[0]-1,], av_patterns[cb[1]-1,])
        diff_patterns[i,] = np.abs((x - x.mean()) / x.std()) 
    
    diff_vec = np.mean(diff_patterns, axis = 0)
    return(diff_vec > threshold)
        
        