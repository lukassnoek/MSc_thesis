# -*- coding: utf-8 -*-
"""
     """

# Importing neccesary packages
import os
import cPickle
import numpy as np
import glob
import glm2mvpa as g2m # previously created module
import matplotlib.pyplot as plt

from sklearn.metrics.pairwise import euclidean_distances as euc_dist

'''
Create some random data set with two factors:
factor_let (A, B) and factor_num (1, 2) with each 15 instances.
Initilialize this as a mvpa_mat object (see glm2mvpa module).
'''

data = np.random.normal(10, 5, (60, 100))
subject_name = 'test'
mask_name = 'random_gauss'
mask_index = None
mask_shape = None
factor_let = ['a'] * 30 + ['b'] * 30
factor_num = [1] * 15 + [2] * 15 + [1] * 15 + [2] * 15
class_labels = zip(factor_let, factor_num)

test_RDM = g2m.mvpa_mat(data, subject_name, mask_name, mask_index, mask_shape, class_labels)

def create_RDM(mvpa_data):
    ''' docstring '''
    
    dist_mat = euc_dist(mvpa_data.data)
    dist_mat = (dist_mat - np.mean(dist_mat)) / np.std(dist_mat)
    np.fill_diagonal(dist_mat, 0)
    return(dist_mat)
        
def create_rdm_regressor(mvpa_data):
    ''' docstring '''
    
    n_fact = len(mvpa_data.class_labels[0])
    
    pred_RDM = np.ones((n_fact + 1,mvpa_data.n_trials,mvpa_data.n_trials))
    
    for fact in xrange(n_fact):    
        entries = zip(*mvpa_data.class_labels)[fact]    

        for idx,trial in enumerate(entries):
            same = [trial == x for x in entries]
            pred_RDM[fact,idx,:] = same
          
    zipped = [str(x) + '_' + str(y) for x,y in mvpa_data.class_labels]
    for idx,trial in enumerate(zipped):
            same = [trial == x for x in zipped]
            pred_RDM[fact+1,idx,:] = same
            
    fig = plt.figure()
    for plot in xrange(n_fact + 1):
        plt.subplot(1, n_fact+1, plot+1)
        plt.imshow(pred_RDM[plot,:,:])
        if plot != n_fact:
            plt.title('Factor ' + str(plot+1))
        else:
            plt.title('Interaction')
   