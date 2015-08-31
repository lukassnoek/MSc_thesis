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
from sklearn import svm
import itertools
from os.path import join as opj

import numpy as np
import nibabel as nib
import h5py
from sklearn.cross_validation import StratifiedShuffleSplit, StratifiedKFold
import pandas as pd
from sklearn.base import TransformerMixin
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from scipy.signal import gaussian

class SelectAboveZvalue(TransformerMixin):
    """ Selects features based on normalized differentation scores above cutoff

    Feature selection method based on largest univariate differences;
    similar to sklearn's univariate feature selection (with fclassif), but
    selects all (normalized) difference scores, which are computed as the
    mean absolute difference between feature values averaged across classes*,
    above a certain z-value.
    * e.g. mean(abs(mean(A)-mean(B), mean(A)-mean(C), mean(B)-mean(C)))

    Works for any amount of classes.

    Attributes:
        zvalue (float/int): cutoff/lowerbound for normalized diff score

    """

    def __init__(self, zvalue):
        self.zvalue = zvalue
        self.idx = None
        self.zvalues = None

    def fit(self, X, y):
        """ Performs feature selection on array of n_samples, n_features
        Args:
            X: array with n_samples, n_features
            y: labels for n_samples
        """

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
            diff_patterns[i,] = np.abs((x - x.mean()) / x.std())

        mean_diff = np.mean(diff_patterns, axis=0)
        self.idx = mean_diff > self.zvalue
        self.zvalues = mean_diff

    def transform(self, X):
        """ Transforms n_samples, n_features array """
        return X[:, self.idx]


def mvp_classify(sub_dir, inputs):
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

    # Unpacking arguments
    iterations = inputs['iterations']
    n_test = inputs['n_test']
    mask_file = inputs['mask_file']
    fs_method = inputs['fs_method']
    fs_arg = inputs['fs_arg']
    fs_average = inputs['fs_average']
    fs_cluster = inputs['fs_cluster']
    cluster_min = inputs['cluster_min']
    cluster_cleanup = inputs['cluster_cleanup']
    cv_method = inputs['cv_method']
    score_unit = 0 if inputs['score_unit'] == 'TPR' else 1
    score_method = inputs['score_method']
    do_pca = inputs['do_pca']

    # Tests
    test_demean_clust = inputs['test_demean_clust']
    test_demean_roi = inputs['test_demean_roi']
    test_demean_gs = inputs['test_demean_gs']

    # Definition of classifier
    clf = svm.LinearSVC()

    # Unpacking subject-data into header and actual data
    header_path, data_path = sub_dir
    mvp = cPickle.load(open(header_path))
    gm_data = h5py.File(data_path, 'r')['data'][:]

    if mvp.subject_name == 'HWW_004':
        print "Performing classification analysis with the following params: "
        for key, value in inputs.iteritems():
            print key + ': ' + str(value)

    # Perform PCA on data (if specified)
    if do_pca:
        print "Performing PCA on data ..."
        pca = PCA()
        gm_data = pca.fit_transform(gm_data)

    # Check feasibility of parameter settings
    if fs_average and len(mask_file) == 1:
        msg = "fs_average with one ROI is impossible."
        raise ValueError(msg)

    #if fs_cluster and len(mask_file) > 1 and not test_demean_roi:
    #    msg = "fs_cluster within individual ROIs is not a good idea."
    #    raise ValueError(msg)

    if fs_cluster and fs_average and not test_demean_roi:
        msg = "fs_average and fs_cluster together doesn't make sense."
        raise ValueError(msg)

    if not fs_cluster and cluster_cleanup:
        msg = "Cannot cleanup clusters without clustering."
        raise ValueError(msg)

    if test_demean_gs:
        print 'demeaning global signal'
        for trial in xrange(gm_data.shape[0]):
            gm_data[trial, :] = gm_data[trial, :] - np.mean(gm_data[trial, :])

    # Start preprocessing data and updating parameters if necessary
    if fs_average:
        print "Averaging features within ROIs ..."

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

        av_idx = av_idx.astype(bool)
        mvp.data = av_data
        mvp.n_features = len(masks)
        mvp.mask_name = 'averaged ROIs'
        mask_file = ['averaged']
        fs_arg = fs_arg if fs_arg < 2 else 0.5

    # If only one mask is specified, we assume that it is all graymatter
    elif len(mask_file) == 1:
        mvp.data = gm_data
        mvp.submask_index = mvp.mask_index[mvp.mask_index]
        mask_file = ['Graymatter mask']

    # If more masks are specified, we will iterate over ROIs
    else:
        print "Iterating over multiple ROIs ..."

    df_list = []  # stores info per mask

    # Start loop over ROIs/masks
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

            #for trial in xrange(mvp.data.shape[0]):
            #    mvp.data[trial, :] = mvp.data[trial, :] - np.mean(mvp.data[trial, :])

        print "Processing %s for subject %s (%i/%i)" % \
              (mvp.mask_name, mvp.subject_name, cMask + 1, len(mask_file))

        n_features = np.sum(mvp.mask_index)

        # Containers for classification data, tracked over iterations
        fs_data = {'count': np.zeros(n_features),
                   'score': np.zeros(n_features)}
        vox_score = np.zeros(n_features)
        conf_mat = np.zeros((mvp.n_class, mvp.n_class))
        feature_count = np.zeros(iterations)
        feature_prop = np.zeros(iterations)

        # Container with class-assignments per trial
        if score_method == 'trial-based':
            trials_score = np.zeros((mvp.n_trials, mvp.n_class))

        # If clustering of features, create cluster_count variable to track
        if fs_cluster:
            cluster_count = np.zeros(iterations)

        # Define cross-validation method with appropriate params
        if cv_method.__name__ == 'StratifiedShuffleSplit':
            folds = cv_method(mvp.num_labels, iterations, n_test * mvp.n_class,
                              random_state=0)
        else:
            folds = cv_method(mvp.num_labels, iterations, random_state=0)

        """ START CROSS-VALIDATION LOOP """
        for i, (train_idx, test_idx) in enumerate(folds):
            print "iteration ", str(i+1)
            # If interactive debugging, uncomment the following lines
            #test_idx = np.array([1, 45, 90])
            #train_idx = np.delete(np.array(range(len(mvp.num_labels))), test_idx)

            # Index data (X) and labels (y)
            train_data = mvp.data[train_idx, :]
            test_data = mvp.data[test_idx, :]
            train_labels = np.asarray(mvp.num_labels)[train_idx]
            test_labels = np.asarray(mvp.num_labels)[test_idx]

            # selector = GenericUnivariateSelect(f_classif, mode=fs_method, param=fs_arg)
            # Define 'selector' with corresponding arg (for feature selection)
            selector = fs_method(fs_arg)
            selector.fit(train_data, train_labels)

            # Vox_idx is a feature-selection-independent index of which voxels
            # are selected (regardless of whether it is contained in clusters,
            # ROIs, etc.)
            vox_idx = np.zeros(vox_score.shape)

            # Cluster feature selection, if specified
            if fs_cluster:
                inpt = {'mvp': mvp, 'train_data': train_data, 'train_labels': train_labels,
                        'test_data': test_data, 'fs_arg': fs_arg,
                        'cluster_min': cluster_min, 'selector': selector,
                        'fs_data': fs_data, 'vox_idx': vox_idx,
                        'cluster_cleanup': cluster_cleanup,
                        'test_demean_clust': test_demean_clust}

                # Cluster data & return averaged (if not cluster_cleanup) ftrs
                output = clustercorrect_feature_selection(**inpt)
                train_data, test_data, cl_idx = output
                cluster_count[i] = train_data.shape[1]

            # if working with averaged ROIs, update params appropriately
            # Note: fs_average and fs_cluster are mutually exclusive params
            elif fs_average:
                for k in xrange(av_idx.shape[1]):
                    if selector.idx[k]:
                        fs_data['score'][av_idx[:, k]] += selector.zvalues[k]
                        fs_data['count'][av_idx[:, k]] += 1
                        vox_idx[av_idx[:, k]] += 1

            # If not fs_cluster or fs_average, simply transform the data
            # with the selection (use only features > fs_arg)
            else:
                train_data = selector.transform(train_data)
                test_data = selector.transform(test_data)
                fs_data['score'][selector.idx] += selector.zvalues[selector.idx]
                fs_data['count'][selector.idx] += 1
                vox_idx[selector.idx] += 1

            # Fit the SVM and store prediction
            clf.fit(train_data, train_labels)
            test_pred = clf.predict(test_data)

            # Update confusion matrix
            conf_mat += confusion_matrix(test_labels, test_pred)

            # If trial-based scoring, update trial_score with class-assignments
            if score_method == 'trial-based':
                trials_score[test_idx, (test_pred - 1).astype(int)] += 1

            # Update score per voxel and feature descriptive statistics
            vox_score[vox_idx.astype(bool)] += clf.score(test_data, test_labels)
            feature_count[i] = train_data.shape[1]
            feature_prop[i] = np.true_divide(feature_count[i], mvp.n_features)

        ### END iteration loop >> write out results of mask/ROI ###

        # Normalize by confusion matrix by the sum of rows (TPR) or sum of
        # columns (PPV)
        conf_mat = np.true_divide(conf_mat, np.sum(conf_mat, score_unit))
        if score_method == 'iteration-based':
            final_score = np.mean(np.diag(conf_mat))

        # If trial-based, trial classification is based on the max. of class-
        # assignments (if trial x is classified once as A, once as B, and twice
        # as C, it is classified as C).
        elif score_method == 'trial-based':
            filter_trials = np.sum(trials_score, 1) == 0
            trials_max = np.argmax(trials_score, 1) + 1
            trials_max[filter_trials] = 0
            trials_predicted = trials_max[trials_max > 0]
            trials_true = np.array(mvp.num_labels)[trials_max > 0]
            conf_mat2 = confusion_matrix(trials_true, trials_predicted)
            conf_mat2 = conf_mat2 / np.sum(conf_mat2, score_unit)
            final_score = np.mean(np.diag(conf_mat2))

        # Calculate mean feature selection score and set NaN to 0
        fs_data['score'] = np.true_divide(fs_data['score'], fs_data['count'])
        fs_data['score'][np.isnan(fs_data['score'])] = 0
        vox_score = np.true_divide(vox_score, fs_data['count'])
        vox_score[np.isnan(vox_score)] = 0

        # Write out feature selection as nifti
        fs_out = np.zeros(mvp.mask_shape).ravel()
        fs_out[mvp.mask_index] = fs_data['score']
        fs_out = fs_out.reshape(mvp.mask_shape)
        fs_img = nib.Nifti1Image(fs_out, np.eye(4))
        file_name = opj(os.getcwd(), '%s_fs.nii.gz' % mvp.subject_name)
        nib.save(fs_img, file_name)

        # Write out voxel scores
        vox_out = np.zeros(mvp.mask_shape).ravel()
        vox_out[mvp.mask_index] = vox_score
        vox_out = vox_out.reshape(mvp.mask_shape)
        vox_img = nib.Nifti1Image(vox_out, np.eye(4))
        file_name = opj(os.getcwd(), '%s_vox.nii.gz' % mvp.subject_name)
        nib.save(vox_img, file_name)

        # Write out classification results as pandas dataframe
        df = {'sub_name': mvp.subject_name,
              'mask': mvp.mask_name,
              'score': np.round(final_score, 3),
              'fs_count': np.round(np.mean(feature_count), 3),
              'fs_prop': np.round(np.mean(feature_prop), 3)}

        if fs_cluster:
            df['cluster_count'] = np.mean(cluster_count)

        df = pd.DataFrame(df, index=[0])

        df_list.append(df)

    df = pd.concat(df_list)
    with open('results_%s.csv' % mvp.subject_name, 'w') as f:
        df.to_csv(f, header=True, sep='\t', index=False)


def clustercorrect_feature_selection(**input):
    """ Performs minimal cluster-thresholding for univariate feature selection

    Args:
        input (dict): information about train/test data/labels and feature
            selection; most important key/value pairs are:
        cluster_min (int): desired minimum cluster size (lower bound)
        cluster_cleanup (bool): if true, perform cluster-thresholding but
            without averaging features within clusters
        test_demean_clust (bool): test to see what happens if the clusters
            are 'demeaned' (weighted linearly by their t-value, which is
            necessary to account for subclusters/smoothing artifacts)

    Returns:
        train_data (ndarray): updated train_data
        test_data (ndarray): updated test_data
        cl_idx (ndarray): indices (boolean) per cluster
        fs_data (dict): updated feature selection data
        vox_idx (ndarray): updated voxel selection indices)
    """

    mvp = input['mvp']
    train_data = input['train_data']
    train_labels = input['train_labels']
    test_data = input['test_data']
    selector = input['selector']
    cluster_min = input['cluster_min']
    fs_arg = input['fs_arg']
    vox_idx = input['vox_idx']
    fs_data = input['fs_data']
    cluster_cleanup = input['cluster_cleanup']
    test_demean_clust = input['test_demean_clust']

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

    cl_idx = np.zeros((mvp.data.shape[1], len(cluster_IDs)))

    for j, clt in enumerate(cluster_IDs):
        idx = (clustered == clt).ravel()[mvp.mask_index]

        if np.sum(idx) < cluster_min:
            break
        else:
            cl_idx[:, j] = idx
    cl_idx = cl_idx[:, np.sum(cl_idx, 0) > 0].astype(bool)
    n_clust = cl_idx.shape[1]

    if test_demean_clust:
        print "demeaning clusters"
        allclust_idx = np.sum(cl_idx, 1).astype(bool)

        for j in xrange(n_clust):
            idx = cl_idx[:, j]

            for t in xrange(train_data.shape[0]):
                #weighting = np.arange(0, 1, np.sum(idx))**2
                xi = np.arange(0,np.sum(idx),1)
                A = np.array([xi, np.ones(np.sum(idx))])
                w = np.linalg.lstsq(A.T,train_data[t,idx])[0] # obtaining the parameters
                line = w[0]*xi+w[1] # regression line
                train_data[t,idx] = np.sort(train_data[t,idx]) - np.sort(train_data[t,idx])*line
                train_data[t,idx] = train_data[t,idx]-np.mean(train_data[t,idx])

            #for t in xrange(test_data.shape[0]):
                #weighting = np.arange(0, 1, np.sum(idx))
                #xi = np.arange(0,np.sum(idx),1)
                #A = np.array([xi, np.ones(np.sum(idx))])
                #w = np.linalg.lstsq(A.T,test_data[t,idx])[0] # obtaining the parameters
                #test_data[t,idx] = np.sort(test_data[t,idx]) - np.sort(test_data[t,idx])*line
                #test_data[t,idx] = test_data[t,idx]-np.mean(test_data[t,idx])

        train_data = train_data[:, allclust_idx]
        test_data = test_data[:, allclust_idx]
        selector.fit(train_data, train_labels)
        fs_data['score'][allclust_idx] += selector.zvalues
        fs_data['count'][allclust_idx] += 1
        vox_idx[allclust_idx] += 1

    elif cluster_cleanup:
        print "cleaning up clusters, not averaging"
        allclust_idx = np.sum(cl_idx, 1).astype(bool)
        train_data = train_data[:, allclust_idx]
        test_data = test_data[:, allclust_idx]
        fs_data['score'][allclust_idx] += selector.zvalues[allclust_idx]
        fs_data['count'][allclust_idx] += 1
        vox_idx[allclust_idx] += 1
    else:
        print "performing regular clustering (averaging)"
        cl_train = np.zeros((train_data.shape[0], cl_idx.shape[1]))
        cl_test = np.zeros((test_data.shape[0], cl_idx.shape[1]))

        for j in xrange(n_clust):
            idx = cl_idx[:, j]
            cl_train[:, j] = np.mean(train_data[:, idx], axis=1)
            cl_test[:, j] = np.mean(test_data[:, idx], axis=1)
            av_cluster = np.mean(selector.zvalues[idx])
            fs_data['score'][idx] += av_cluster
            fs_data['count'][idx] += 1
            vox_idx[idx] += 1
        train_data = cl_train
        test_data = cl_test

    mvp.n_features = train_data.shape[1]
    output = (train_data, test_data, cl_idx, fs_data, vox_idx)

    return output


def average_classification_results(inputs):
    """ Averages results across subjects """
    iterations = inputs['iterations']
    n_test = inputs['n_test']
    mask_file = inputs['mask_file']
    fs_method = inputs['fs_method']
    fs_arg = inputs['fs_arg']
    fs_average = inputs['fs_average']
    fs_cluster = inputs['fs_cluster']
    cluster_min = inputs['cluster_min']
    test_demean_clust = inputs['test_demean_clust']
    cv_method = inputs['cv_method']

    # Create header
    fid = open('analysis_parameters', 'w')
    fid.write('Classification run with the following parameters: \n \n')

    if len(mask_file) == 1:
        fid.write('Mask \t %s \n' % mask_file)

    fid.write('Iterations: \t %s \n' % iterations)
    fid.write('CV_method: \t %s \n' % cv_method.__name__)
    fid.write('N-test: \t %s \n' % n_test)
    fid.write('fs_method: \t %s \n' % fs_method.__name__)
    fid.write('fs_arg: \t %f \n' % fs_arg)
    fid.write('fs_average: \t %s \n' % fs_average)
    fid.write('fs_cluster: \t %s \n' % fs_cluster)

    if fs_cluster:
        fid.write('cluster_min: \t %i \n' % cluster_min)

    fid.write('test_demean_clust: \t %s \n \n' % test_demean_clust)
    fid.close()

    to_load = glob.glob('*results_HWW*.csv')

    dfs = []
    for sub in to_load:
        dfs.append(pd.DataFrame.from_csv(sub, sep='\t'))

    dfs = pd.concat(dfs)
    [os.remove(p) for p in glob.glob('*.csv')]

    to_write = {}
    df_list = []
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
        av_score = np.mean(dfs['score']).round(3)

        with open('analysis_parameters', 'a') as f:
            dfs.to_csv(f, header=True, sep='\t', index=False)
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

    fs_av = np.true_divide(fs_sum, len(zipped_files))
    img = nib.Nifti1Image(fs_av, np.eye(4))
    file_name = opj(os.getcwd(), 'averaged_feature_selection.nii.gz')
    nib.save(img, file_name)

    img = nib.Nifti1Image(vox_sum, np.eye(4))
    file_name = opj(os.getcwd(), 'averaged_voxel_accuracy.nii.gz')
    nib.save(img, file_name)

    os.system('cat %s' % filename)


if __name__ == '__main__':
    sys.path.append('/home/c6386806/LOCAL/Analysis_scripts')

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
    inputs = {}
    inputs['iterations'] = 10
    inputs['n_test'] = 4
    #inputs['mask_file'] = sorted(glob.glob(opj(ROI_dir, 'Harvard_Oxford_atlas', 'bilateral', '*nii.gz*')))
    #inputs['mask_file'] = sorted(glob.glob(opj(ROI_dir, 'Harvard_Oxford_atlas', 'unilateral', '*nii.gz*')))
    inputs['mask_file'] = glob.glob(opj(ROI_dir, 'GrayMatter.nii.gz'))
    inputs['fs_method'] = SelectAboveZvalue
    inputs['fs_arg'] = 1.5
    inputs['fs_average'] = False
    inputs['fs_cluster'] = True
    inputs['cluster_min'] = 200
    inputs['cluster_cleanup'] = False
    inputs['test_demean_clust'] = True
    inputs['test_demean_roi'] = False
    inputs['test_demean_gs'] = False
    inputs['cv_method'] = StratifiedShuffleSplit
    inputs['score_unit'] = 'PPV'
    inputs['score_method'] = 'iteration-based'  # trial-based
    inputs['do_pca'] = False

    debug = False
    n_proc = 1 if debug else len(subject_dirs)

    # Run classification on n_cores = len(subjects)
    Parallel(n_jobs=n_proc) \
        (delayed(mvp_classify)(sub_dir, inputs) for sub_dir in subject_dirs)
    sub_dir = subject_dirs[0]
    # mvp_classify(sub_dir, inputs)
    average_classification_results(inputs)
