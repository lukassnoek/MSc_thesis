# -*- coding: utf-8 -*-
"""
Main classification module

To do:
- separate mvpa_mat into header (cPickle) and data (HDF5)

Lukas Snoek
"""

_author_ = "Lukas Snoek"

import glob
import os
import numpy as np
import itertools
import cPickle
import nibabel as nib

from sklearn import svm

from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt

def draw_random_subsample(mvpa, n_test):
    ''' 
    Draws random subsample of test trials for each class.
    Assumes the mvpa_mat() object as input.
    
    Args:
        mvpa (mvpa_mat): instance of mvpa_mat (see glm2mvpa module)
        n_test (int): the number of test-trial to be drawn PER CLASS
    
    Returns:
        test_ind (bool): boolean vector of len(mvpa.n_trials)
    '''
    
    test_ind = np.zeros(len(mvpa.num_labels), dtype=np.int)
    
    
    for c in xrange(mvpa.n_class):
        ind = np.random.choice(mvpa.trial_idx[c], n_test, replace=False)            
        test_ind[ind] = 1
    return test_ind.astype(bool)


def select_voxels(mvpa, train_idx, zval, method):
    ''' 
    Feature selection based on univariate differences between
    patterns averaged across trials from the same class. 
    
    Args:
        mvpa (mvpa_mat): instance of mvpa_mat (see glm2mvpa module)
        train_idx (bool): boolean vector with train-trials (inverse of test_idx)
        zval (float/int): z-value cutoff/threshold for normalized pattern differences
        method (str): method to select differences; 'pairwise' = pairwise univar.
                      differences; 'fstat' = differences averaged across classes.
                      
    Returns:
        feat_idx (bool): index with selected features of len(mvpa.n_features)
        diff_vec (ndarray): array with actual difference scores
    '''
   
    # Setting useful parameters & pre-allocation
    data = mvpa.data[train_idx,] 
    av_patterns = np.zeros((mvpa.n_class, mvpa.n_features))
    
    # Calculate mean patterns
    for c in xrange(mvpa.n_class):         
        av_patterns[c,:] = np.mean(data[mvpa.class_idx[c][train_idx],:], axis = 0)
       
    # Create difference vectors, z-score standardization, absolute
    comb = list(itertools.combinations(range(1,mvpa.n_class+1),2))
    diff_patterns = np.zeros((len(comb), mvpa.n_features))
        
    for i, cb in enumerate(comb):        
        x = av_patterns[cb[0]-1] - av_patterns[cb[1]-1,:]
        diff_patterns[i,] = np.abs((x - x.mean()) / x.std()) 
    
    if diff_patterns.shape[0] > 1 and method == 'fstat':
        diff_vec = np.mean(diff_patterns, axis = 0) 
        feat_idx = diff_vec > zval 
    elif diff_patterns.shape[0] > 1 and method == 'pairwise':
        diff_vec = np.mean(diff_patterns, axis = 0)        
        feat_idx = np.sum(diff_patterns > zval, axis = 0) > 0       
    else:
        diff_vec = diff_patterns
        feat_idx = diff_vec > zval
    
    return(feat_idx, diff_vec)

def mvpa_classify(identifier, mask_file, iterations, n_test, zval, method):
    '''
    Main classification function that classifies subject-specific
    mvpa_mat objects according to their classes (specified in .num_labels).
    Very rudimentary version atm.
    
    Args:
        identifier (str): string to be matched when globbing cPickle-files         
        iterations (int): amount of cross-validation iterations
        n_test (int): amount of test-trials per iteration (see select_voxels)
        zval (float): z-value cutoff score (see select_voxels)
        method (str): method to determine univariate differences between mean
                      patterns across classes.
    '''

    subject_dirs = glob.glob(os.path.join(os.getcwd(),'mvpa_mats','*%s*cPickle' % identifier))

    # We need to load the first sub to extract some info
    print "Processing file %s..." % os.path.basename(subject_dirs[0])
    mvpa = cPickle.load(open(subject_dirs[0]))
    
    clf = svm.LinearSVC()
        
    for c, sub_dir in enumerate(subject_dirs):
        
        if sub_dir is not subject_dirs[0]:
            print "Processing file %s..." % os.path.basename(sub_dir)
            mvpa = cPickle.load(open(sub_dir))

        mask_data = np.reshape(nib.load(mask_file).get_data() > 10, mvpa.mask_index.shape)
        new_idx = ((mask_data.astype(int) + mvpa.mask_index.astype(int))==2)[mvpa.mask_index]

        mvpa.mask_name = os.path.basename(mask_file)[:-7]
        mvpa.mask_index = new_idx
        mvpa.data = mvpa.data[:,new_idx]
        mvpa.n_features = np.sum(new_idx)

        score = []
        for i in xrange(iterations):
            test_idx = draw_random_subsample(mvpa, n_test)
            train_idx = np.invert(test_idx)
            
            feat_idx,diff_vec = select_voxels(mvpa,train_idx,zval,method = 'pairwise')
    
            feat_idx_dd,diff_vec_dd = select_voxels(mvpa,np.ones(len(mvpa.num_labels)).astype(bool),zval,method = 'pairwise')            
            
            out = np.zeros(mvpa.mask_index.shape)        
            out[mvpa.mask_index] = diff_vec_dd
            out = np.reshape(out, mvpa.mask_shape)            
            
            out[out < zval] = 0            
            img = nib.Nifti1Image(out, np.eye(4))
            file_name = os.path.join(os.getcwd(),'voxel_selection.nii.gz')
            nib.save(img, file_name)

            #os.system('cluster -i %s -o clustered -t %f -osize=cluster_size > cluster_info' %
            #         (file_name, zval))


            if np.sum(feat_idx) == 0:
                 raise ValueError('Z-threshold too high! No voxels selected ' \
                 + 'at iteration ' + str(i) + ' for ' + mvpa.subject_name)
            
            train_data = mvpa.data[train_idx,:][:,feat_idx]
            train_data_dd = mvpa.data[train_idx,:][:,feat_idx_dd]
            
            test_data = mvpa.data[test_idx,:][:,feat_idx]
            test_data_dd = mvpa.data[test_idx,:][:,feat_idx_dd]
            
            #plt.imshow(np.corrcoef(train_data_dd),interpolation='none')

            train_labels = np.asarray(mvpa.num_labels)[train_idx]
            test_labels = np.asarray(mvpa.num_labels)[test_idx]
            
            model = clf.fit(train_data, train_labels)
            score_train = clf.score(train_data, train_labels)
            score_test = clf.score(test_data, test_labels)            
            
            model = clf.fit(train_data_dd, train_labels)
            score_test_dd = clf.score(test_data_dd, test_labels) 
            
            score.append(np.mean(score_test))
            
        print "Score subject %i: %f" % (c+1,np.mean(score))
'''
        for cls in range(mvpa.n_class):
            voxels_correct[c,:,cls] = voxels_correct[c,:,cls] / (voxels_selected[c,:] * n_test)
        
        print 'Done processing ' + sub_dir
        
            
    maxfilt = np.sum(trials_predicted, axis = 2) == 0
    maxpred = np.argmax(trials_predicted, axis = 2) + 1   
    maxpred[maxfilt] = 0
    
    # Create confusion matrix
    cm_all = maxpred2cm(maxpred, n_sub, mvpa)
    plot_confusion_matrix(np.mean(cm_all, axis = 0), mvpa, \
                          title = 'Confusion matrix, averaged over subs')
    
    
    voxels_correct[np.isnan(voxels_correct)] = 0
    voxels_correct = np.mean(voxels_correct, axis = 2)    
    print "Averaged classification score: " + str(np.mean(np.diag(np.mean(cm_all, 0))))
    '''
    #return(cm_all, voxels_correct)
    
def plot_confusion_matrix(cm, mvpa, title='Confusion matrix', cmap=plt.cm.Reds):
    ''' Code from sklearn's example at http://scikit-learn.org/stable/
    auto_examples/model_selection/plot_confusion_matrix.html '''
    
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(mvpa.class_names))
    plt.xticks(tick_marks, [x[0:3] for x in mvpa.class_names], rotation=45)
    plt.yticks(tick_marks, [x[0:3] for x in mvpa.class_names])
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def maxpred2cm(maxpred, n_sub, mvpa):
    cm_all = np.zeros((n_sub, mvpa.n_class, mvpa.n_class))
    for c in xrange(n_sub):
        for row in xrange(mvpa.n_class):
            for col in xrange(mvpa.n_class):
                cm_all[c,row,col] = np.sum(maxpred[c,mvpa.class_idx[row]] == col+1)    
                
        # Normalize
        cm_all[c,:,:] = cm_all[c,:,:] / np.sum(cm_all[c,:,:], axis = 0)
    
    return(cm_all)