# -*- coding: utf-8 -*-
"""
Main classification module

Lukas Snoek
"""

import glob
import os
import numpy as np
import itertools
import cPickle
from sklearn import svm
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt

def draw_random_subsample(mvpa, n_train):
    ''' Draws random subsample of test trials for each class '''
    test_ind = np.zeros(len(mvpa.num_labels), dtype=np.int)
    
    for c in xrange(mvpa.n_class):
        ind = np.random.choice(mvpa.class_idx[c], n_train, replace=False)            
        test_ind[ind-1] = 1
        
    return test_ind.astype(bool)
    
def select_voxels(mvpa, train_ind, threshold):
    ''' 
    Feature selection. Runs on single subject data
    '''
    
    # Setting useful parameters & pre-allocation
    n_train = np.sum(train_ind) / mvpa.n_class
    trial_idx = [range(n_train*x,n_train*(x+1)) for x in range(mvpa.n_class)]
    data = mvpa.data[train_ind,] # should be moved to main_classify() later
    av_patterns = np.zeros((mvpa.n_class, mvpa.n_features))
    
    # Calculate mean patterns
    for c in xrange(mvpa.n_class):         
        av_patterns[c,] = np.mean(data[trial_idx[c],], axis = 0)
       
    # Create difference vectors, z-score standardization, absolute
    comb = list(itertools.combinations(range(1,mvpa.n_class+1),2))
    diff_patterns = np.zeros((len(comb), mvpa.n_features))
        
    for i, cb in enumerate(comb):        
        x = np.subtract(av_patterns[cb[0]-1,], av_patterns[cb[1]-1,])
        diff_patterns[i,] = np.abs((x - x.mean()) / x.std()) 
    
    diff_vec = np.mean(diff_patterns, axis = 0)
    return(diff_vec > threshold)
   
def mvpa_classify(iterations, n_test):
    
    subject_dirs = glob.glob(os.getcwd() + '/*cPickle')      
    n_sub = len(subject_dirs)      
    
    # We need to load the first sub to extract some info
    mvpa = cPickle.load(open(subject_dirs[0]))
    trials_selected = np.zeros((n_sub, mvpa.n_trials))     
    trials_predicted = np.zeros((n_sub, mvpa.n_trials, mvpa.n_class))
    
    for c, sub_dir in enumerate(subject_dirs):
        
        if sub_dir is not subject_dirs[0]:
            mvpa = cPickle.load(open(sub_dir))
    
        for i in xrange(iterations):
            test_idx = draw_random_subsample(mvpa, n_test)
            train_idx = np.invert(test_idx)
            feat_idx = select_voxels(mvpa,train_idx,1.5)

            if np.sum(feat_idx) == 0:
                 raise ValueError('Z-threshold too high! No voxels selected.')
                            
            train_data = mvpa.data[train_idx,:][:,feat_idx]
            test_data = mvpa.data[test_idx,:][:,feat_idx]
        
            clf = svm.SVC()
            clf.fit(train_data, mvpa.num_labels[train_idx])
            x = clf.predict(test_data)
            
            # Update trials_selected and trials_predicted
            trials_selected[c,test_idx] += 1            
            for i in range(len(x)):
                trials_predicted[c,test_idx,x-1] += 1
    
    maxfilt = np.sum(trials_predicted, axis = 2) == 0
    maxpred = np.argmax(trials_predicted, axis = 2) + 1   
    maxpred[maxfilt] = 0
    
    # Create confusion matrix
    cm_all = np.zeros((n_sub, mvpa.n_class, mvpa.n_class))
    for c in xrange(n_sub):
        j = 0
    
        for row in xrange(mvpa.n_class):
            for col in xrange(mvpa.n_class):
                cm_all[c,row,col] = np.sum(maxpred[c,range(j,j+mvpa.n_inst)] == col+1)
            j += mvpa.n_inst
        cm_all[c,:,:] = cm_all[c,:,:] / np.sum(cm_all[c,:,:], axis = 0)
    
    plot_confusion_matrix(np.mean(cm_all, axis = 0), mvpa, \
                          title = 'Confusion matrix, averaged over subs')
    
    print "Averaged classification score: " + str(np.mean(np.diag(np.mean(cm_all, 0))))
    
    return(cm_all)
    
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
