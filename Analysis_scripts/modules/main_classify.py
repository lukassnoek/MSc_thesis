# -*- coding: utf-8 -*-
"""
Main classification module

Lukas Snoek
"""

__author__ = "Lukas Snoek"

import sys
import glob
import os
import sys
import numpy as np
import itertools
import cPickle
import nibabel as nib
import h5py
import datetime
import time
import pandas as pd
from sklearn import svm
from sklearn.cross_validation import StratifiedShuffleSplit
import multiprocessing as mp
import pandas as pd

from sklearn.feature_selection import f_classif, GenericUnivariateSelect
from os.path import join as opj
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
import sklearn.feature_selection

from sklearn.base import TransformerMixin
import itertools

class SelectAboveZvalue(TransformerMixin):
    """ DOESNT WORK YET """

    def __init__(self, zvalue, idx=None):
        self.zvalue = zvalue
        self.idx = None

    def fit(self, X, y):
        n_class = np.unique(y).shape[0]
        n_features = X.shape[1]

        av_patterns = np.zeros((n_class, n_features))

        # Calculate mean patterns
        for i in xrange(n_class):
            av_patterns[i, :] = np.mean(X[y == np.unique(y)[i], :], axis=0)

            # Create difference vectors, z-score standardization, absolute
            comb = list(itertools.combinations(range(1, n_class + 1), 2))
            diff_patterns = np.zeros((len(comb), n_features))

        for i, cb in enumerate(comb):
            x = av_patterns[cb[0] - 1] - av_patterns[cb[1] - 1, :]
            diff_patterns[i, ] = np.abs((x - x.mean()) / x.std())

        self.idx = np.mean(diff_patterns, axis=0) > self.zvalue

    def transform(self, X):
        return X[:, self.idx]

def mvp_classify(sub_dir, mask_file, iterations, n_test, fs_method, fs_arg):
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
    gm_data = h5py.File(data_path, 'r')['data'][:]

    df_list = []

    # Re-indexing with ROI
    for cMask, roi in enumerate(mask_file):

        if cMask == 'GrayMatter.nii.gz':
            pass
        else:
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

        print "Processing %s for subject %s (%i/%i)" % (mvp.mask_name, mvp.subject_name,
                                                          cMask + 1, len(mask_file))

        sss = StratifiedShuffleSplit(y=mvp.num_labels, n_iter=iterations, test_size=n_test * mvp.n_class, random_state=0)

        score = np.zeros(iterations)
        for i, (train_idx, test_idx) in enumerate(sss):

            train_data = mvp.data[train_idx, :]
            test_data = mvp.data[test_idx, :]

            train_labels = np.asarray(mvp.num_labels)[train_idx]
            test_labels = np.asarray(mvp.num_labels)[test_idx]

            #selector = GenericUnivariateSelect(f_classif, mode=fs_method, param=fs_arg)
            selector = fs_method(fs_arg)

            #out = np.zeros(mvp.mask_shape).ravel()
            #out[mvp.mask_index][new_idx] = diff_vec_dd
            #out = np.reshape(out, mvp.mask_shape)

            #out[out < zval] = 0
            #img = nib.Nifti1Image(out, np.eye(4))
            #file_name = opj(os.getcwd(), 'voxel_selection.nii.gz')
            #nib.save(img, file_name)

            # os.system('cluster -i %s -o clustered -t %f -osize=cluster_size > cluster_info' %
            #         (file_name, zval))
            selector.fit(train_data, train_labels)

            clf.fit(selector.transform(train_data), train_labels)
            score[i] = clf.score(selector.transform(test_data), test_labels)

        df = {'sub_name': mvp.subject_name,
              'mask': mvp.mask_name,
              'score': np.mean(score)}
        df = pd.DataFrame(df, index=[1])

        df_list.append(df)

    return(pd.concat(df_list))


if __name__ == "__main__":

    from joblib import Parallel, delayed

    sys.path.append('/home/c6386806/LOCAL/Analysis_scripts/')
    home = os.path.expanduser("~")
    feat_dir = opj(home, 'DynamicAffect_MV', 'FSL_FirstLevel_StimulusDriven')
    ROI_dir = opj(home, 'ROIs')
    os.chdir(feat_dir)

    # Params
    identifier = ''
    iterations = 100
    n_test = 4
    zvalue = 2.3
    mask_file = sorted(glob.glob(opj(ROI_dir, 'GrayMatter.nii.gz')))
    from sklearn.feature_selection import SelectFdr
    fs_method = SelectAboveZvalue
    fs_arg = 2.3

    mvp_dir = opj(os.getcwd(), 'mvp_mats')
    header_dirs = sorted(glob.glob(opj(mvp_dir, '*%s*cPickle' % identifier)))
    data_dirs = sorted(glob.glob(opj(mvp_dir, '*%s*hdf5' % identifier)))
    subject_dirs = zip(header_dirs, data_dirs)

    results = Parallel(n_jobs=len(subject_dirs)) \
        (delayed(mvp_classify)(sub_dir, mask_file, iterations, n_test, fs_method, 2.3) for sub_dir in subject_dirs)

    results = pd.concat(results)
    masks = np.unique(results['mask'])

    total_df = []
    for mask in masks:
        score = np.mean(results['score'][results['mask'] == mask]).round(3)
        df = {'mask': mask,
              'score': score}

        total_df.append(pd.DataFrame(df, index=[1]))

    to_write = pd.concat(total_df)

    ts = time.time()
    now = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d_%H:%M')

    fid = open('results_summary', 'w')
    fid.write('## Results classification run with the following parameters \n \n')
    fid.write('# Iterations: \t %s \n' % iterations)
    fid.write('# N-test: \t %s \n' % n_test)
    fid.write('# ')
    fid.write('# Mask_name \t \t Mean score \n')
    fid.close()

    with open('results_summary', 'a') as f:
        to_write.to_csv(f, header=False, sep='\t', index=False)