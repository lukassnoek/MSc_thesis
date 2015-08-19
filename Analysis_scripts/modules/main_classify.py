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
import cPickle
import nibabel as nib
import h5py
import datetime
import time
from sklearn import svm
from sklearn.cross_validation import StratifiedShuffleSplit
import pandas as pd
import itertools

from sklearn.feature_selection import f_classif, GenericUnivariateSelect
from os.path import join as opj
from sklearn.metrics import confusion_matrix
from sklearn.base import TransformerMixin
from nipype.interfaces.fsl.model import Cluster as cluster

class SelectAboveZvalue(TransformerMixin):
    """
    <<docstring>>
    """

    def __init__(self, zvalue):
        self.zvalue = zvalue
        self.idx = None
        self.zvalues = None

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

        mean_diff = np.mean(diff_patterns, axis=0)
        self.idx = mean_diff > self.zvalue
        self.zvalues = mean_diff

    def transform(self, X):
        return X[:, self.idx]

def mvp_classify(sub_dir, mask_file, iterations, n_test, fs_method, fs_arg, fs_average, fs_cluster, cluster_min):
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

    if fs_average and len(mask_file) > 1:
        print "Averaging features within ROIs ..."
        mvp = average_features_within_rois(mvp, gm_data, mask_file)
        mask_file = ['averaged']

    elif len(mask_file) == 1 and os.path.basename(mask_file[0]) == 'GrayMatter.nii.gz':
        mvp.data = gm_data
        mvp.submask_index = mvp.mask_index[mvp.mask_index]
        mask_file = ['GrayMatter only']

    for cMask, roi in enumerate(mask_file):

        if len(mask_file) > 1:
            mask_data = nib.load(roi).get_data()
            mask_mask = mask_data > mvp.mask_threshold
            mask_mask = np.reshape(mask_mask, mvp.mask_index.shape)
            mask_overlap = mask_mask.astype(int) + mvp.mask_index.astype(int)

            # Upload MVPHeader
            mvp.submask_index = (mask_overlap == 2)[mvp.mask_index]
            mvp.mask_name = os.path.basename(roi)[:-7]
            mvp.data = gm_data[:, mvp.submask_index]
            mvp.n_features = np.sum(mvp.submask_index)

            """
            Possibility: use multivariate prewhitening on patterns
            res4d = nib.load(sub_path + '/stats_new/res4d_mni.nii.gz').get_data()
            res4d.resize([np.prod(res4d.shape[0:3]), res4d.shape[3]])
            res4d = res4d[mask_index,]
            pattern * inv(sqrt(res4d))???
            """

        print "Processing %s for subject %s (%i/%i)" % (mvp.mask_name, mvp.subject_name,
                                                          cMask + 1, len(mask_file))

        fs_count = np.zeros(mvp.n_features)
        fs_score = np.zeros(mvp.n_features)

        sss = StratifiedShuffleSplit(y=mvp.num_labels, n_iter=iterations,
                                     test_size=n_test * mvp.n_class, random_state=0)

        score = np.zeros(iterations)
        for i, (train_idx, test_idx) in enumerate(sss):
            train_data = mvp.data[train_idx, :]
            test_data = mvp.data[test_idx, :]

            train_labels = np.asarray(mvp.num_labels)
            train_labels = np.asarray(mvp.num_labels)[train_idx]
            test_labels = np.asarray(mvp.num_labels)[test_idx]

            #selector = GenericUnivariateSelect(f_classif, mode=fs_method, param=fs_arg)
            selector = fs_method(fs_arg)

            # os.system('cluster -i %s -o clustered -t %f -osize=cluster_size > cluster_info' %
            #         (file_name, zval))
            selector.fit(train_data, train_labels)

            if fs_cluster:
                fs = np.zeros(mvp.mask_shape).ravel()
                fs[mvp.mask_index] = selector.zvalues
                fs = fs.reshape(mvp.mask_shape)
                img = nib.Nifti1Image(fs, np.eye(4))
                file_name = opj(os.getcwd(), '%s_ToCluster.nii.gz' % mvp.subject_name)
                nib.save(img, file_name)

                cmd = 'cluster -i %s -t %f -o %s --no_table' % (file_name, fs_arg, file_name)
                _ = os.system(cmd)

                clustered = nib.load(file_name).get_data()
                clt_idc = sorted(np.unique(clustered), reverse=True)

                cl_train_data = np.zeros((train_data.shape[0], len(clt_idc)))
                cl_test_data = np.zeros((test_data.shape[0], len(clt_idc)))

                for j, clt in enumerate(clt_idc):
                    idx = (clustered == clt).ravel()[mvp.mask_index]

                    if np.sum(idx) < cluster_min:
                        break
                    else:
                        cl_train_data[:, j] = np.mean(train_data[:, idx], axis=1)
                        cl_test_data[:, j] = np.mean(test_data[:, idx], axis=1)

                train_data = cl_train_data[np.invert(cl_train_data == 0)]
                train_data = train_data.reshape((cl_train_data.shape[0], train_data.shape[0] / cl_train_data.shape[0]))
                test_data = cl_test_data[np.invert(cl_test_data == 0)]
                test_data = test_data.reshape((cl_test_data.shape[0], test_data.shape[0] / cl_test_data.shape[0]))

            fs_score[selector.idx] = fs_score[selector.idx] + selector.zvalues[selector.idx]
            fs_count[selector.idx] += 1

            clf.fit(train_data, train_labels)
            score[i] = clf.score(test_data, test_labels)

        fs_score = np.divide(fs_score, fs_count)
        fs_score[np.isnan(fs_score)] = 0

        #out = np.zeros(mvp.mask_shape).ravel()
        #out[mvp.mask_index] = fs_score
        #out = out.reshape(mvp.mask_shape)

        #img = nib.Nifti1Image(out, np.eye(4))
        #file_name = opj(os.getcwd(), '%s_feature_selection.nii.gz' % mvp.subject_name)
        #nib.save(img, file_name)

        df = {'sub_name': mvp.subject_name,
              #'contrast': mvp.c
              'mask': mvp.mask_name,
              'score': np.mean(score)}
        df = pd.DataFrame(df, index=[1])

        df_list.append(df)

    return(pd.concat(df_list))


def average_features_within_rois(mvp, gm_data, masks):

    av_data = np.zeros((mvp.n_trials, len(masks)))

    for cMask, roi in enumerate(masks):
        mask_data = nib.load(roi).get_data()
        mask_mask = mask_data > mvp.mask_threshold
        mask_mask = np.reshape(mask_mask, mvp.mask_index.shape)
        mask_overlap = mask_mask.astype(int) + mvp.mask_index.astype(int)
        idx = (mask_overlap == 2)[mvp.mask_index]
        av_data[:, cMask] = np.mean(gm_data[:, idx], axis=1)

    mvp.data = av_data
    mvp.n_features = len(mask_file)

    return(mvp)

def average_classification_results(sub_results):
    sub_results = pd.concat(sub_results)
    masks = np.unique(sub_results['mask'])

    total_df = []
    for mask in masks:
        score = np.mean(sub_results['score'][sub_results['mask'] == mask]).round(3)
        df = {'mask': mask,
              'score': score}


        total_df.append(pd.DataFrame(df, index=[1]))

    # Write out results
    to_write = pd.concat(total_df)
    ts = time.time()
    now = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d_%H:%M')

    fid = open('results_summary', 'w')
    fid.write('## Results classification run with the following parameters \n \n')
    fid.write('# Iterations: \t %s \n' % iterations)
    fid.write('# N-test: \t %s \n' % n_test)
    fid.write('# fs_method: \t %s \n' % fs_method)
    fid.write('# fs_arg: \t %f \n' % fs_arg)
    fid.write('# Mask_name \t \t Mean score \n')
    fid.close()

    with open('results_summary', 'a') as f:
        to_write.to_csv(f, header=False, sep='\t', index=False)

    to_average = glob.glob('*.nii.gz')
    mni_mask = np.zeros((91,109,91))

    for f in to_average:
        data = nib.load(f).get_data()
        mni_mask = mni_mask + data

    mni_mask = np.divide(mni_mask, len(to_average))
    img = nib.Nifti1Image(mni_mask, np.eye(4))
    file_name = opj(os.getcwd(), 'averaged_feature_selection.nii.gz')
    nib.save(img, file_name)

    os.system('cat results_summary')


if __name__ == "__main__":

    sys.path.append('/home/c6386806/LOCAL/Analysis_scripts/')

    from joblib import Parallel, delayed
    from sklearn.feature_selection import SelectFdr
    from modules.glm2mvpa import MVPHeader

    # Information about which data to use
    home = os.path.expanduser('~')
    feat_dir = opj(home, 'DecodingEmotions')
    ROI_dir = opj(home, 'ROIs')
    os.chdir(feat_dir)
    identifier = ''

    mvp_dir = opj(os.getcwd(), 'mvp_mats')
    header_dirs = sorted(glob.glob(opj(mvp_dir, '*%s*cPickle' % identifier)))
    data_dirs = sorted(glob.glob(opj(mvp_dir, '*%s*hdf5' % identifier)))
    subject_dirs = zip(header_dirs, data_dirs)

    # Parameters for classification
    iterations = 100
    n_test = 1
    #mask_file = sorted(glob.glob(opj(ROI_dir, 'Harvard_Oxford_atlas', 'unilateral', '*nii.gz*')))
    mask_file = glob.glob(opj(ROI_dir, 'GrayMatter.nii.gz'))
    fs_method = SelectAboveZvalue
    fs_arg = 1.5
    fs_average = False
    fs_cluster = True
    cluster_min = 150

    # Run classification on n_cores = len(subjects)
    sub_results = Parallel(n_jobs=len(subject_dirs)) \
        (delayed(mvp_classify)(sub_dir, mask_file, iterations, n_test, fs_method,
                               fs_arg, fs_average, fs_cluster, cluster_min) for sub_dir in subject_dirs)

    average_classification_results(sub_results)
    # Concatenate subject-specific results and extract mean correct per mask
