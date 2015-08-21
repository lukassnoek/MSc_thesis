# -*- coding: utf-8 -*-
"""
Main classification module

Lukas Snoek
"""
from __future__ import division

__author__ = "Lukas Snoek"

import glob
import os
import sys
import cPickle
import datetime
import time
from sklearn import svm
import itertools
from os.path import join as opj

import numpy as np
import nibabel as nib
import h5py
from sklearn.cross_validation import StratifiedShuffleSplit
import pandas as pd
from sklearn.base import TransformerMixin
from sklearn.metrics import confusion_matrix


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


def mvp_classify(sub_dir, mask_file, iterations, n_test, fs_method, fs_arg,
                 fs_average, fs_cluster, cluster_min, test_demean):
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

    # Definition of classifier
    clf = svm.LinearSVC()

    # Unpacking subject-data into header and actual data
    header_path, data_path = sub_dir
    mvp = cPickle.load(open(header_path))
    gm_data = h5py.File(data_path, 'r')['data'][:]

    df_list = []

    # Check feasibility of parameter settings
    if fs_average and len(mask_file) == 1:
        msg = "fs_average with one ROI is impossible ... abort."
        raise ValueError(msg)

    if fs_cluster and len(mask_file) > 1:
        msg = "fs_cluster within individual ROIs is not a good idea ... abort."
        raise ValueError(msg)

    if fs_cluster and fs_average:
        msg = "fs_average and fs_cluster together doesn't make sense ... abort."
        raise ValueError(msg)

    # Start preprocessing data and updating parameters if necessary
    if fs_average:
        print "Averaging features within ROIs ..."
        mvp, av_idx = average_features_within_rois(mvp, gm_data, mask_file)
        mask_file = ['averaged']

        """
        if test_demean:
            print "Demeaning features ..."
            for r in xrange(mvp.data.shape[1]):
                av_idx = av_idx.astype(bool)
                print mvp.data.shape
                gm_data[:, av_idx[:, r]] -= mvp.data[:, r, np.newaxis]

            mvp.data = gm_data
            mvp.n_features = mvp.data.shape[1]
            mask_file = ['GrayMatter only, demean test']
            mvp.mask_name = mask_file[0]
        """

    elif len(mask_file) == 1:
        print "Working with entire Graymatter mask ..."
        mvp.data = gm_data
        mvp.submask_index = mvp.mask_index[mvp.mask_index]
        mask_file = ['Graymatter mask']
    else:
        print "Iterating over multiple ROIs ..."

    # Start loop over ROIs <<if len(mask_file) > 1>>
    for cMask, roi in enumerate(mask_file):

        # Index data with roi (mask)
        if len(mask_file) > 1:
            mask_mask = nib.load(roi).get_data() > mvp.mask_threshold
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

        print "Processing %s for subject %s (%i/%i)" % \
              (mvp.mask_name, mvp.subject_name, cMask + 1, len(mask_file))

        # Containers for classification data
        gm_features = np.sum(mvp.mask_index)

        score = np.zeros(iterations)
        fs_data = {'count': np.zeros(gm_features),
                   'score': np.zeros(gm_features)}
        vox_score = np.zeros(gm_features)
        conf_mat = np.zeros((mvp.n_class, mvp.n_class))
        feature_count = np.zeros(iterations)
        feature_prop = np.zeros(iterations)

        if fs_cluster:
            cluster_count = np.zeros(iterations)

        ### Create cross_validation scheme and start iteration-loop ###
        sss = StratifiedShuffleSplit(mvp.num_labels, iterations,
                                     n_test * mvp.n_class, random_state=0)

        for i, (train_idx, test_idx) in enumerate(sss):

            # Index data (X) and labels (y)
            train_data = mvp.data[train_idx, :]
            test_data = mvp.data[test_idx, :]
            train_labels = np.asarray(mvp.num_labels)[train_idx]
            test_labels = np.asarray(mvp.num_labels)[test_idx]

            #selector = GenericUnivariateSelect(f_classif, mode=fs_method, param=fs_arg)
            # Define 'selector' with corresponding arg (for feature selection)
            selector = fs_method(fs_arg)
            selector.fit(train_data, train_labels)

            vox_idx = np.zeros(vox_score.shape)

            # Cluster feature selection, if specified
            if fs_cluster:
                inpt = {'mvp': mvp, 'train_data': train_data,
                        'test_data': test_data, 'fs_arg': fs_arg,
                        'cluster_min': cluster_min, 'selector': selector}

                output = clustercorrect_feature_selection(**inpt)
                mvp, train_data, test_data, cl_idx, fs = output

                cluster_count[i] = cl_idx.shape[1]

                # Update fs_data with average cluster score
                for k in xrange(cl_idx.shape[1]):
                    av_cluster = np.mean(fs.ravel()[mvp.mask_index][cl_idx[:, k]])
                    fs_data['score'][cl_idx[:, k]] += av_cluster
                    fs_data['count'][cl_idx[:, k]] += 1
                    vox_idx[cl_idx[:, k]] += 1

                """
                if test_demean:
                    print "demeaning features"
                    for r in xrange(cl_idx.shape[1]):
                        cl_idx = cl_idx.astype(bool)
                        gm_data[train_idx,:][:, cl_idx[:, r]] -= train_data[:, r, np.newaxis]
                        gm_data[test_idx,:][:, cl_idx[:, r]] -= test_data[:, r, np.newaxis]

                    mvp.data = gm_data
                    mvp.n_features = mvp.data.shape[1]
                    train_data = mvp.data[train_idx, :]
                    test_data = mvp.data[test_idx, :]

                    train_data = selector.transform(train_data)
                    test_data = selector.transform(test_data)
                    idx = selector.idx
                    fs_score[idx] += selector.zvalues[idx]
                    fs_count[idx] += 1
                """
            elif fs_average:
                # If features are averaged within ROIs, update appropriately
                for k in xrange(av_idx.shape[1]):
                    if selector.idx[k]:
                        fs_data['score'][av_idx[:, k]] += selector.zvalues[k]
                        fs_data['count'][av_idx[:, k]] += 1
                        vox_idx[av_idx[:, k]] += 1
            else:
                # If not fs_cluster or fs_average, simply transform the data
                # with the selection (use only features > fs_arg (prob. zvalue)
                train_data = selector.transform(train_data)
                test_data = selector.transform(test_data)
                fs_data['score'][selector.idx] += selector.zvalues[selector.idx]
                fs_data['count'][selector.idx] += 1
                vox_idx[selector.idx] += 1

            # Fit the SVM and score!
            clf.fit(train_data, train_labels)
            test_pred = clf.predict(test_data)
            conf_mat += confusion_matrix(test_labels, test_pred)
            score[i] = clf.score(test_data, test_labels)

            vox_score[vox_idx.astype(bool)] += score[i]
            feature_count[i] = train_data.shape[1]
            feature_prop[i] = np.divide(feature_count[i], mvp.n_features)

        ### END iteration loop >> write out results of mask/ROI ###

        conf_mat = np.divide(conf_mat, iterations * n_test)

        # Calculate mean feature selection score and set NaN to 0
        fs_data['score'] = np.divide(fs_data['score'], fs_data['count'])
        fs_data['score'][np.isnan(fs_data['score'])] = 0
        vox_score = np.divide(vox_score, fs_data['count'])
        vox_score[np.isnan(vox_score)] = 0

        # Write out feature selection as nifti
        fs_out = np.zeros(mvp.mask_shape).ravel()
        fs_out[mvp.mask_index] = fs_data['score']
        fs_out = fs_out.reshape(mvp.mask_shape)

        fs_img = nib.Nifti1Image(fs_out, np.eye(4))
        file_name = opj(os.getcwd(), '%s_feature_selection.nii.gz' % mvp.subject_name)
        nib.save(fs_img, file_name)

        vox_out = np.zeros(mvp.mask_shape).ravel()
        vox_out[mvp.mask_index] = vox_score
        vox_out = vox_out.reshape(mvp.mask_shape)

        vox_img = nib.Nifti1Image(vox_out, np.eye(4))
        file_name = opj(os.getcwd(), '%s_voxel_accuracy.nii.gz' % mvp.subject_name)
        nib.save(vox_img, file_name)

        # Write out classification results as pandas dataframe
        df = {'sub_name': mvp.subject_name,
              'mask': mvp.mask_name,
              'score': np.round(np.mean(score), 3),
              'fs_count': np.round(np.mean(feature_count), 3),
              'fs_prop': np.round(np.mean(feature_prop), 3)}

        if fs_cluster:
            df['cluster_count'] = np.mean(cluster_count)

        df = pd.DataFrame(df, index=[0])

        df_list.append(df)

    df = pd.concat(df_list)
    with open('results_%s.csv' % mvp.subject_name, 'w') as f:
        df.to_csv(f, header=True, sep='\t', index=False)


def average_features_within_rois(mvp, gm_data, masks):

    av_data = np.zeros((mvp.n_trials, len(masks)))
    av_idx = np.zeros((mvp.n_features, len(masks)))

    for cMask, roi in enumerate(masks):
        mask_data = nib.load(roi).get_data()
        mask_mask = mask_data > mvp.mask_threshold
        mask_mask = np.reshape(mask_mask, mvp.mask_index.shape)
        mask_overlap = mask_mask.astype(int) + mvp.mask_index.astype(int)
        idx = (mask_overlap == 2)[mvp.mask_index]

        av_data[:, cMask] = np.mean(gm_data[:, idx], axis=1)
        av_idx[:, cMask] = idx

    mvp.data = av_data
    mvp.n_features = len(mask_file)
    mvp.mask_name = 'averaged ROIs'

    return mvp, av_idx.astype(bool)


def clustercorrect_feature_selection(**input):

    mvp = input['mvp']
    train_data = input['train_data']
    test_data = input['test_data']
    selector = input['selector']
    cluster_min = input['cluster_min']

    fs = np.zeros(mvp.mask_shape).ravel()
    fs[mvp.mask_index] = selector.zvalues
    fs = fs.reshape(mvp.mask_shape)
    img = nib.Nifti1Image(fs, np.eye(4))
    file_name = opj(os.getcwd(), '%s_ToCluster.nii.gz' % mvp.subject_name)
    nib.save(img, file_name)

    cmd = 'cluster -i %s -t %f -o %s --no_table' % (file_name, fs_arg, file_name)
    _ = os.system(cmd)

    clustered = nib.load(file_name).get_data()
    cluster_IDs = sorted(np.unique(clustered), reverse=True)

    cl_train = np.zeros((train_data.shape[0], len(cluster_IDs)))
    cl_test = np.zeros((test_data.shape[0], len(cluster_IDs)))
    cl_idx = np.zeros((mvp.data.shape[1], len(cluster_IDs)))

    for j, clt in enumerate(cluster_IDs):
        idx = (clustered == clt).ravel()[mvp.mask_index]

        if np.sum(idx) < cluster_min:
            break
        else:
            cl_idx[:, j] = idx
            cl_train[:, j] = np.mean(train_data[:, idx], axis=1)
            cl_test[:, j] = np.mean(test_data[:, idx], axis=1)

    train_data = cl_train[:, np.invert((np.sum(cl_train, 0)) == 0)]
    test_data = cl_test[:, np.invert((np.sum(cl_test, 0)) == 0)]
    cl_idx = cl_idx[:, np.sum(cl_idx, 0) > 0].astype(bool)

    mvp.n_features = train_data.shape[1]
    output = (mvp, train_data, test_data, cl_idx, fs)

    return output


def average_classification_results(args):

    mask_file, fs_method, fs_arg, fs_average, fs_cluster, cluster_min, test_demean = args

    # Create header
    fid = open('analysis_parameters', 'w')
    fid.write('Classification run with the following parameters: \n \n')

    if len(mask_file) == 1:
        fid.write('Mask \t %s \n' % mask_file)

    fid.write('Iterations: \t %s \n' % iterations)
    fid.write('N-test: \t %s \n' % n_test)
    fid.write('fs_method: \t %s \n' % fs_method)
    fid.write('fs_arg: \t %f \n' % fs_arg)
    fid.write('fs_average: \t %s \n' % fs_average)
    fid.write('fs_cluster: \t %s \n' % fs_cluster)

    if fs_cluster:
        fid.write('cluster_min: \t %i \n' % cluster_min)

    fid.write('test_demean: \t %f \n \n' % test_demean)
    fid.close()

    to_load = glob.glob('*results*.csv')

    dfs = []
    for sub in to_load:
        dfs.append(pd.DataFrame.from_csv(sub, sep='\t'))

    dfs = pd.concat(dfs)
    [os.remove(p) for p in glob.glob('*.csv')]

    to_write = {}
    df_list =[]
    masks = np.unique(dfs['mask'])

    if len(masks) > 1:

        for mask in masks:
            to_write['mask'] = mask
            to_write['score'] = np.mean(dfs['score'][dfs['mask'] == mask])
            df_list.append(pd.DataFrame(to_write, index=[0]))

        df_list = pd.concat(df_list)
        with open('analysis_parameters', 'a') as f:
            df_list.to_csv(f, header=True, sep='\t', index=False)

        filename = 'results_per_mask.csv'
        os.rename('analysis_parameters', filename)

    else:
        subs = np.unique(dfs['sub_name'])

        for s in subs:
            to_write['subject'] = s
            to_write['score'] = np.mean(dfs['score'][dfs['sub_name'] == s])
            df_list.append(pd.DataFrame(to_write, index=[0]))

        df_list = pd.concat(df_list)
        av_score = np.mean(df_list['score']).round(3)

        with open('analysis_parameters', 'a') as f:
            df_list.to_csv(f, header=True, sep='\t', index=False)
            f.write('\n Average score: \t %f' % av_score)

        filename = 'results_per_sub.csv'
        os.rename('analysis_parameters', filename)

    [os.remove(p) for p in glob.glob('*ToCluster*')]
    [os.remove(p) for p in glob.glob('*averaged*')]

    fs_files = glob.glob('*feature_selection*')
    vox_files = glob.glob('*voxel_accuracy*')
    zipped_files = zip(fs_files, vox_files)

    fs_sum = np.zeros((91, 109, 91))
    vox_sum = np.zeros((91, 109, 91))

    for fs, vox in zipped_files:
        fs_sub = nib.load(fs).get_data()
        fs_sum += fs_sub

        vox_sub = nib.load(vox).get_data()
        vox_sum += vox_sub > 0.4

    fs_av = np.divide(fs_sum, len(zipped_files))
    img = nib.Nifti1Image(fs_av, np.eye(4))
    file_name = opj(os.getcwd(), 'averaged_feature_selection.nii.gz')
    nib.save(img, file_name)

    img = nib.Nifti1Image(vox_sum, np.eye(4))
    file_name = opj(os.getcwd(), 'averaged_voxel_accuracy.nii.gz')
    nib.save(img, file_name)

    os.system('cat %s' % filename)


if __name__ == "__main__":

    sys.path.append('/home/c6386806/LOCAL/Analysis_scripts/')

    from joblib import Parallel, delayed
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
    iterations = 10
    n_test = 1
    mask_file = sorted(glob.glob(opj(ROI_dir, 'Harvard_Oxford_atlas', 'bilateral', '*nii.gz*')))
    #mask_file = glob.glob(opj(ROI_dir, 'GrayMatter.nii.gz'))
    fs_method = SelectAboveZvalue
    fs_arg = 1
    fs_average = False
    fs_cluster = False
    cluster_min = 10
    test_demean = False

    # Run classification on n_cores = len(subjects)
    Parallel(n_jobs=len(subject_dirs)) \
        (delayed(mvp_classify)(sub_dir, mask_file, iterations, n_test, fs_method,
                               fs_arg, fs_average, fs_cluster, cluster_min, test_demean) for sub_dir in subject_dirs)

    args = (mask_file, fs_method, fs_arg, fs_average, fs_cluster, cluster_min,
            test_demean)
    average_classification_results(args)
