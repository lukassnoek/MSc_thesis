# -*- coding: utf-8 -*-
"""
Main classification module

Lukas Snoek
"""

_author_ = "Lukas Snoek"

import glob
import os
import numpy as np
import itertools
import cPickle
import nibabel as nib
import h5py
import datetime
import time
import pandas as pd
from sklearn import svm
from os.path import join as opj

from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt


def draw_random_subsample(mvp, n_test):
    """
    Draws random subsample of test trials for each class.
    Assumes the mvp_mat() object as input.

    Args:
        mvp (mvp_mat): instance of mvp_mat (see glm2mvp module)
        n_test (int): the number of test-trial to be drawn PER CLASS

    Returns:
        test_ind (bool): boolean vector of len(mvp.n_trials)
    """

    test_ind = np.zeros(len(mvp.num_labels), dtype=np.int)

    for c in xrange(mvp.n_class):
        ind = np.random.choice(mvp.trial_idx[c], n_test, replace=False)
        test_ind[ind] = 1
    return test_ind.astype(bool)


def select_voxels(mvp, train_idx, zval, vs_method):
    ''' 
    Feature selection based on univariate differences between
    patterns averaged across trials from the same class. 
    
    Args:
        mvp (mvp_mat): instance of mvp_mat (see glm2mvp module)
        train_idx (bool): boolean vector with train-trials (inverse of test_idx)
        zval (float/int): z-value cutoff/threshold for normalized pattern differences
        method (str): method to select differences; 'pairwise' = pairwise univar.
                      differences; 'fstat' = differences averaged across classes.
                      
    Returns:
        feat_idx (bool): index with selected features of len(mvp.n_features)
        diff_vec (ndarray): array with actual difference scores
    '''

    # Setting useful parameters & pre-allocation
    data = mvp.data[train_idx,]
    av_patterns = np.zeros((mvp.n_class, mvp.n_features))

    # Calculate mean patterns
    for c in xrange(mvp.n_class):
        av_patterns[c, :] = np.mean(data[mvp.class_idx[c][train_idx], :],
                                    axis=0)

    # Create difference vectors, z-score standardization, absolute
    comb = list(itertools.combinations(range(1, mvp.n_class + 1), 2))
    diff_patterns = np.zeros((len(comb), mvp.n_features))

    for i, cb in enumerate(comb):
        x = av_patterns[cb[0] - 1] - av_patterns[cb[1] - 1, :]
        diff_patterns[i,] = np.abs((x - x.mean()) / x.std())

    if diff_patterns.shape[0] > 1 and vs_method == 'fstat':
        diff_vec = np.mean(diff_patterns, axis=0)
        feat_idx = diff_vec > zval
    elif diff_patterns.shape[0] > 1 and vs_method == 'pairwise':
        diff_vec = np.mean(diff_patterns, axis=0)
        feat_idx = np.sum(diff_patterns > zval, axis=0) > 0
    else:
        diff_vec = diff_patterns
        feat_idx = diff_vec > zval

    return (np.squeeze(feat_idx), diff_vec)


def mvp_classify(sub_dir, mask_file, iterations, n_test, zval, vs_method):
    """
    Main classification function that classifies subject-specific
    mvp_mat objects according to their classes (specified in .num_labels).
    Very rudimentary version atm.
    
    Args:
        identifier (str): string to be matched when globbing cPickle-files         
        iterations (int): amount of cross-validation iterations
        n_test (int): amount of test-trials per iteration (see select_voxels)
        zval (float): z-value cutoff score (see select_voxels)
        method (str): method to determine univariate differences between mean
                      patterns across classes.
    """

    clf = svm.LinearSVC()

    header_path, data_path = sub_dir

    mvp = cPickle.load(open(header_path))
    h5f = h5py.File(data_path, 'r')
    gm_data = h5f['data'][:]

    df_list = []

    # Re-indexing with ROI
    for roi in mask_file:
        mask_data = nib.load(roi).get_data()
        mask_data = np.reshape(mask_data > mvp.mask_threshold,
                               mvp.mask_index.shape)
        new_idx = ((mask_data.astype(int) +
                    mvp.mask_index.astype(int)) == 2)[mvp.mask_index]
        mvp.mask_name = os.path.basename(roi)[:-7]
        mvp.data = gm_data[:, new_idx]
        mvp.n_features = np.sum(new_idx)

        #res4d = nib.load(sub_path + '/stats_new/res4d_mni.nii.gz').get_data()
        #res4d.resize([np.prod(res4d.shape[0:3]), res4d.shape[3]])
        #res4d = res4d[mask_index,]

        print "Processing %s for subject %s" % (mvp.mask_name, mvp.subject_name)

        score = np.zeros(iterations)
        for i in xrange(iterations):

            test_idx = draw_random_subsample(mvp, n_test)
            train_idx = np.invert(test_idx)

            feat_idx, diff_vec = select_voxels(mvp, train_idx,
                                               zval, vs_method=vs_method)

        #out = np.zeros(mvp.mask_shape).ravel()
        #out[mvp.mask_index][new_idx] = diff_vec_dd
        #out = np.reshape(out, mvp.mask_shape)

        #out[out < zval] = 0
        #img = nib.Nifti1Image(out, np.eye(4))
        #file_name = opj(os.getcwd(), 'voxel_selection.nii.gz')
        #nib.save(img, file_name)

        # os.system('cluster -i %s -o clustered -t %f -osize=cluster_size > cluster_info' %
        #         (file_name, zval))

            train_data = mvp.data[train_idx, :][:, feat_idx]
            test_data = mvp.data[test_idx, :][:, feat_idx]

            train_labels = np.asarray(mvp.num_labels)[train_idx]
            test_labels = np.asarray(mvp.num_labels)[test_idx]

            clf.fit(train_data, train_labels)
            score[i] = clf.score(test_data, test_labels)

        df = {'sub_name': mvp.subject_name,
              'mask': mvp.mask_name,
              'score': np.mean(score)}
        df = pd.DataFrame(df, index=[1])

        df_list.append(df)

    return(pd.concat(df_list))

'''
        for cls in range(mvp.n_class):
            voxels_correct[c,:,cls] = voxels_correct[c,:,cls] / (voxels_selected[c,:] * n_test)
        
        print 'Done processing ' + sub_dir
        
            
    maxfilt = np.sum(trials_predicted, axis = 2) == 0
    maxpred = np.argmax(trials_predicted, axis = 2) + 1   
    maxpred[maxfilt] = 0
    
    # Create confusion matrix
    cm_all = maxpred2cm(maxpred, n_sub, mvp)
    plot_confusion_matrix(np.mean(cm_all, axis = 0), mvp, \
                          title = 'Confusion matrix, averaged over subs')
    
    
    voxels_correct[np.isnan(voxels_correct)] = 0
    voxels_correct = np.mean(voxels_correct, axis = 2)    
    print "Averaged classification score: " + str(np.mean(np.diag(np.mean(cm_all, 0))))
    '''


# return(cm_all, voxels_correct)


def create_results_log(iterations, zval, n_test, vs_method):

    ts = time.time()
    now = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d_%H:%M')

    fid = open('results_summary', 'w')
    fid.write('## Results classification run with the following parameters \n \n')
    fid.write('# Iterations: \t %s \n' % iterations)
    fid.write('# Z-value: \t %s \n' % zval)
    fid.write('# N-test: \t %s \n' % n_test)
    fid.write('# Method: \t %s \n \n' % vs_method)

    fid.write('# Mask_name \t \t Mean score \n')
    fid.close()


def maxpred2cm(maxpred, n_sub, mvp):
    cm_all = np.zeros((n_sub, mvp.n_class, mvp.n_class))
    for c in xrange(n_sub):
        for row in xrange(mvp.n_class):
            for col in xrange(mvp.n_class):
                cm_all[c, row, col] = np.sum(
                    maxpred[c, mvp.class_idx[row]] == col + 1)

        # Normalize
        cm_all[c, :, :] = cm_all[c, :, :] / np.sum(cm_all[c, :, :], axis=0)

    return (cm_all)
