# -*- coding: utf-8 -*-
"""
Main classification module

Lukas Snoek
"""

from __future__ import division
import glob
import os
import sys
import cPickle
import csv
import pandas as pd
import itertools as itls
import numpy as np
import nibabel as nib
import h5py
import progressbar as pbar
from os.path import join as opj
from sklearn import svm
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.base import TransformerMixin
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
from sklearn.decomposition import RandomizedPCA
from joblib import Parallel, delayed
from scipy.ndimage.measurements import label
from glm2mvpa import MVPHeader

__author__ = "Lukas Snoek"


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
        comb = list(itls.combinations(range(1, n_class + 1), 2))
        diff_patterns = np.zeros((len(comb), n_features))
        for i, cb in enumerate(comb):
            x = av_patterns[cb[0] - 1] - av_patterns[cb[1] - 1, :]
            diff_patterns[i, :] = np.abs((x - x.mean()) / x.std())

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
    clf = inputs['clf']
    iterations = inputs['iterations']
    n_test = inputs['n_test']
    mask_file = inputs['mask_file']
    fs_method = inputs['fs_method']
    fs_arg = inputs['fs_arg']
    fs_doubledip = inputs['fs_doubledip']
    fs_average = inputs['fs_average']
    cluster_args = inputs['cluster_args']
    pca_args = inputs['pca_args']
    cv_method = inputs['cv_method']
    score_method = inputs['score_method']
    save_feat_corrs = inputs['save_feat_corrs']
    subject_timer = inputs['subject_timer']
    demean_patterns = inputs['demean_patterns']
    average_patterns = inputs['average_patterns']

    # Unpacking subject-data into header and actual data
    header_path, data_path = sub_dir
    mvp = cPickle.load(open(header_path))
    gm_data = h5py.File(data_path, 'r')['data'][:]

    # test with spatial filtering
    '''
    for trial in xrange(gm_data.shape[0]):
        tmp = np.zeros(mvp.mask_shape).ravel()
        tmp[mvp.mask_index] = gm_data[trial, :]
        filt = gaussian_filter(tmp.reshape(mvp.mask_shape), 3)
        gm_data[trial, :] = filt.ravel()[mvp.mask_index].ravel()
    '''

    # Average features within ROIs and return: trials x ROIs matrix (av_data)
    if fs_average:
        print "Averaging features within ROIs ..."

        mvp.data = np.zeros((mvp.n_trials, len(mask_file)))
        av_idx = np.zeros((mvp.n_features, len(mask_file)))

        for cMask, roi in enumerate(mask_file):
            mask_mask = nib.load(roi).get_data() > mvp.mask_threshold
            mask_mask = np.reshape(mask_mask, mvp.mask_index.shape)
            mask_overlap = mask_mask.astype(int) + mvp.mask_index.astype(int)
            idx = (mask_overlap == 2)[mvp.mask_index]

            mvp.data[:, cMask] = np.mean(gm_data[:, idx], axis=1)
            av_idx[:, cMask] = idx

        # Update mvp attributes after averaging within ROIs
        mvp.n_features = len(mask_file)
        mvp.mask_name = 'averaged ROIs'
        mask_file = ['averaged']

    # If only one mask is specified, it is assumed to be a whole-brain
    # graymatter mask, which is by default contained in gm_data
    elif len(mask_file) == 1:
        mvp.data = gm_data
        mask_file = ['Graymatter mask']

    # If more masks are specified, we will iterate over ROIs
    else:
        pass

    # df_list stores results per mask, which is later concatenated
    df_list = []

    # Loop over masks; if only one mask, loop-size = 1
    for cMask, roi in enumerate(mask_file):

        # Index data with roi (mask)
        if len(mask_file) > 1:
            mask_mask = nib.load(roi).get_data() > mvp.mask_threshold
            mask_mask = np.reshape(mask_mask, mvp.mask_index.shape)
            mask_overlap = mask_mask.astype(int) + mvp.mask_index.astype(int)

            # Update MVPHeader
            mvp.submask_index = (mask_overlap == 2)[mvp.mask_index]
            mvp.mask_name = os.path.basename(roi)[:-7]
            mvp.data = gm_data[:, mvp.submask_index]
            mvp.n_features = np.sum(mvp.submask_index)

        # Containers for classification data, tracked over iterations
        # idea for future implementation: create pandas dataframe
        n_features = np.sum(mvp.mask_index)
        fs_data = {'count': np.zeros(n_features), 'score': np.zeros(n_features)}
        vox_score = np.zeros(n_features)
        feature_count = np.zeros(iterations)
        correlations = np.zeros(iterations)
        feats_per_clust = np.zeros(iterations)

    # Container with class-assignments per trial / score metrics
        if score_method == 'trial_based':
            trials_score = np.zeros((mvp.n_trials, mvp.n_class))
        elif score_method == 'iteration_based':
            precision = np.zeros(iterations)
            accuracy = np.zeros(iterations)
            recall = np.zeros(iterations)

        folds = cv_method(mvp.num_labels, iterations, n_test * mvp.n_class, random_state=0)

        # Create progressbar for ONE subject (specified by subject_timer)
        subject_timer = mvp.subject_name if inputs['n_cores'] == 1 else subject_timer

        if mvp.subject_name == subject_timer:
            print "\nProcessing mask: %s (%i out of %i)" % \
                  (mvp.mask_name, cMask + 1, len(mask_file))

            widgets = ['CV loop (%i iterations): ' % len(folds),
                       pbar.Percentage(), ' ', pbar.Bar('>'), ' ', pbar.ETA(), ' ']

            pb = pbar.ProgressBar(widgets=widgets, maxval=len(folds)).start()

        """ START CROSS-VALIDATION LOOP """
        for i, (train_idx, test_idx) in enumerate(folds):
            # Update progressbar
            if mvp.subject_name == subject_timer:
                pb.update(i+1)

            """ If interactive debugging, uncomment the following lines
            test_idx = np.array([1, 45, 90])
            train_idx = np.delete(np.array(range(len(mvp.num_labels))), test_idx)
            """

            # Index data (X) and labels (y); num_labels has range (1, 2, 3)
            train_data = mvp.data[train_idx, :]
            test_data = mvp.data[test_idx, :]
            train_labels = mvp.num_labels[train_idx]
            test_labels = mvp.num_labels[test_idx]

            # Vox_idx is a feature-selection-independent index of which voxels
            # are selected (regardless of whether it is contained in clusters,
            # ROIs, etc.), based on analysis-dependent variable n-features
            vox_idx = np.zeros(vox_score.shape)

            skip_fs = True if average_patterns and fs_arg < 0 or pca_args['do_pca'] else False

            if not skip_fs:
                # Univariate feature selection using selector
                selector = fs_method(fs_arg)

                if fs_doubledip:
                    selector.fit(mvp.data, mvp.num_labels)
                else:
                    selector.fit(train_data, train_labels)
            else:
                fs_arg = -1

            ''' Split of analysis into either clustering procedure (fs_cluster),
            or continuing with ROI-averaged features (fs_average), or with
            all features from a generic univariate feature selection '''

            if cluster_args['do_clust']:

                # This block of code (which refers to the clustercorrect_feature
                # _selection function) can be optimized/written more clearly
                input_cc = {'mvp': mvp, 'train_data': train_data,
                            'train_labels': train_labels, 'test_data': test_data,
                            'cluster_args': cluster_args, 'selector': selector,
                            'fs_data': fs_data, 'vox_idx': vox_idx, }

                # Cluster data & return averaged (if not cluster_cleanup) ftrs
                output = clustercorrect_feature_selection(input_cc)
                train_data, test_data, cl_idx, fs_data, vox_idx = output
                feats_per_clust[i] = np.round(np.sum(vox_idx) / train_data.shape[1])

            # If working with averaged ROIs, update params appropriately
            # Note: fs_average and fs_cluster are mutually exclusive params
            elif fs_average:# If working with averaged ROIs, update params appropriately

                train_data = selector.transform(train_data)
                test_data = selector.transform(test_data)

                for k in xrange(av_idx.shape[1]):
                    if selector.idx[k]:
                        fs_data['score'][av_idx[:, k].astype(bool)] += selector.zvalues[k]
                        fs_data['count'][av_idx[:, k].astype(bool)] += 1
                        vox_idx[av_idx[:, k].astype(bool)] += 1

            elif fs_arg > 0: # implying generic univariate feature selection
                train_data = selector.transform(train_data)
                test_data = selector.transform(test_data)
                fs_data['score'][selector.idx] += selector.zvalues[selector.idx]
                fs_data['count'][selector.idx] += 1
                vox_idx[selector.idx] += 1

            # Perform PCA on data (if specified)
            if pca_args['do_pca']:
                pca = RandomizedPCA(n_components=pca_args['n_comp']+pca_args['start_idx']-1)
                pca.fit(train_data)
                train_data = pca.transform(train_data)[:, pca_args['start_idx']-1:]
                test_data = pca.transform(test_data)[:, pca_args['start_idx']-1:]

            if demean_patterns:
                train_data = train_data - np.expand_dims(train_data.mean(axis=1), axis=1)

            if average_patterns:
                train_data = np.mean(train_data, axis=1)
                test_data = np.mean(test_data, axis=1)

            if train_data.ndim == 1:
                train_data = np.expand_dims(train_data, axis=1)
                test_data = np.expand_dims(test_data, axis=1)

            # Fit the SVM and store prediction
            clf.fit(train_data, train_labels)
            prediction = clf.predict(test_data)

            # Update scores
            if score_method == 'iteration_based':
                precision[i] = precision_score(test_labels, prediction, average='macro')
                recall[i] = recall_score(test_labels, prediction, average='macro')
                accuracy[i] = accuracy_score(test_labels, prediction)
            elif score_method == 'trial_based':
                trials_score[test_idx, (prediction - 1).astype(int)] += 1

            # Update score per voxel and feature descriptive statistics
            vox_score[vox_idx.astype(bool)] += clf.score(test_data, test_labels)
            #vox_score[vox_idx.astype(bool)] += np.mean(clf.coef_, axis=0)
            feature_count[i] = train_data.shape[1]

            save_feat_corrs = False if train_data.shape[1] == 1 else save_feat_corrs
            if save_feat_corrs:
                corrs_tmp = np.corrcoef(train_data.T)
                correlations[i] = np.mean(corrs_tmp[np.tril_indices(np.sqrt(corrs_tmp.size), -1)])

        """ END CROSS-VALIDATION LOOP """
        if mvp.subject_name == subject_timer:
            pb.finish()

        if score_method == 'iteration_based':
            precision = np.round(np.mean(precision), 3)
            recall = np.round(np.mean(recall), 3)
            accuracy = np.round(np.mean(recall), 3)

        # If trial-based, trial classification is based on the max. of class-
        # assignments (if trial x is classified once as A, once as B, and twice
        # as C, it is classified as C).
        elif score_method == 'trial_based':
            filter_trials = np.sum(trials_score, 1) == 0
            trials_max = np.argmax(trials_score, 1) + 1
            trials_max[filter_trials] = 0
            trials_predicted = trials_max[trials_max > 0]
            trials_true = np.array(mvp.num_labels)[trials_max > 0]
            conf_mat2 = confusion_matrix(trials_true, trials_predicted)
            np.save('confmat_%s' % mvp.subject_name, conf_mat2)

            # Calculate scores
            precision = precision_score(trials_true, trials_predicted, average='macro')
            recall = recall_score(trials_true, trials_predicted, average='macro')
            accuracy = accuracy_score(trials_true, trials_predicted)

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
              'accuracy': np.round(accuracy, 3),
              'precision': np.round(precision, 3),
              'recall': np.round(recall, 3),
              'fs_count': np.round(np.mean(feature_count), 3),
              'fs_std': np.round(np.std(feature_count), 3),
              'corr_vox:': np.round(np.mean(correlations), 3)}

        df = pd.DataFrame(df, index=[0])
        df_list.append(df)

    df = pd.concat(df_list)
    with open('subresults_%s.csv' % mvp.subject_name, 'w') as f:
        df.to_csv(f, header=True, sep='\t', index=False)


def clustercorrect_feature_selection(input_cc):
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

    # Unpacking arguments
    mvp = input_cc['mvp']
    train_data = input_cc['train_data']
    train_labels = input_cc['train_labels']
    test_data = input_cc['test_data']
    selector = input_cc['selector']
    cluster_args = input_cc['cluster_args']
    vox_idx = input_cc['vox_idx']
    fs_data = input_cc['fs_data']

    # Create fs: vector of length mask_shape with univariate feature weights
    fs = np.zeros(mvp.mask_shape).ravel()
    fs[mvp.mask_index] = selector.zvalues
    fs = fs.reshape(mvp.mask_shape)

    # Use clustercorrect_generic to perform actual cluster-thresholding
    cl_idx = clustercorrect_generic(fs, mvp, selector.zvalue, cluster_args['minimum'])
    n_clust = cl_idx.shape[1]
    allclust_idx = np.sum(cl_idx, 1).astype(bool)

    """
    Once the cluster-indices have been calculated, three options are possible:
    1. Cluster-threshold (i.e. remove clusters below the minimum cluster-size
       threshold; referred to as 'cluster-cleanup'). Clusters are NOT averaged.
    2. Cluster-threshold + average clusters (default option)
    3. Cluster-threshold + de-mean within clusters +  no averaging
       (test_demean_clust)
    """

    if cluster_args['cleanup']:
        train_data = train_data[:, allclust_idx]
        test_data = test_data[:, allclust_idx]
        fs_data['score'][allclust_idx] += selector.zvalues[allclust_idx]
        fs_data['count'][allclust_idx] += 1
        vox_idx[allclust_idx] += 1

    elif cluster_args['test_demean']:

        # Demean clusters separately
        for j in xrange(n_clust):
            idx = cl_idx[:, j]
            tmp_train = train_data[:, idx]
            tmp_test = test_data[:, idx]
            train_data[:, idx] = tmp_train - np.expand_dims(tmp_train.mean(axis=1), axis=1)
            test_data[:, idx] = tmp_test - np.expand_dims(tmp_test.mean(axis=1), axis=1)

        # Perform feature selection again!
        selector.fit(train_data, train_labels)

        fs = np.zeros(mvp.mask_shape).ravel()
        fs[mvp.mask_index] = selector.zvalues
        fs = fs.reshape(mvp.mask_shape)

        minimum = np.round(1 * cluster_args['minimum'])
        cl_idx = clustercorrect_generic(fs, mvp, selector.zvalue, minimum)
        n_clust = cl_idx.shape[1]
        allclust_idx = np.sum(cl_idx, 1).astype(bool)

    if cluster_args['test_demean'] or not cluster_args['cleanup']:

        cl_train = np.zeros((train_data.shape[0], cl_idx.shape[1]))
        cl_test = np.zeros((test_data.shape[0], cl_idx.shape[1]))

        for j in xrange(n_clust):
            idx = cl_idx[:, j]
            cl_train[:, j] = np.median(train_data[:, idx], axis=1)
            cl_test[:, j] = np.median(test_data[:, idx], axis=1)
            av_cluster = np.mean(selector.zvalues[idx])
            fs_data['score'][idx] += av_cluster
            fs_data['count'][idx] += 1
            vox_idx[idx] += 1

        # Updating train_data (trials X voxels) to clustered (trials X clusters)
        train_data = cl_train
        test_data = cl_test

    # Update n_features attribute to current shape of train_data
    mvp.n_features = train_data.shape[1]
    return train_data, test_data, cl_idx, fs_data, vox_idx


def clustercorrect_generic(data, mvp, zvalue, minimum):

    clustered, num_clust = label(data>zvalue)
    values, counts = np.unique(clustered.ravel(), return_counts=True)
    n_clust = np.argmax(np.sort(counts)[::-1] < minimum)

    # Sort and trim
    cluster_IDs = values[counts.argsort()[::-1][:n_clust]]
    #cluster_nfeat = np.delete(np.sort(counts)[::-1][:n_clust], 0)
    cluster_IDs = np.delete(cluster_IDs, 0)

    # cl_idx holds indices per cluster
    cl_idx = np.zeros((mvp.data.shape[1], len(cluster_IDs)))

    # Update cl_idx until cluster-size < cluster_min
    for j, clt in enumerate(cluster_IDs):
        cl_idx[:, j] = (clustered == clt).ravel()[mvp.mask_index]

    return cl_idx.astype(bool)

def average_classification_results(inputs):
    """ Averages results across subjects"""

    # Create header
    inputs['mask_file'] = 'ROIs' if len(inputs['mask_file']) > 1 else inputs['mask_file']

    with open('analysis_parameters', 'w') as fid:
        fid.write('Classification run with the following parameters: \n \n')
        writer = csv.writer(fid, delimiter='\t')
        writer.writerows(inputs.items())
        fid.write('\n')

    to_load = glob.glob('*subresults_*.csv')

    dfs = []
    for sub in to_load:
        dfs.append(pd.DataFrame(pd.read_csv(sub, sep='\t', index_col=False)))

    dfs = pd.concat(dfs)

    [os.remove(p) for p in glob.glob('*.csv')]

    to_write = {}
    df_list = []
    masks = np.unique(dfs['mask'])

    # If we've iterated over multiple ROIs, report average ROI-score
    if len(masks) > 1:

        for mask in masks:
            to_write['mask'] = mask

            for column in dfs.columns:
                print dfs[column].dtype
                if dfs[column].dtype == np.float64 or dfs[column].dtype == np.int64:
                    to_write[column + '_av'] = np.mean(dfs[column][dfs['mask'] == mask])
                    to_write[column + '_std'] = np.std(dfs[column][dfs['mask'] == mask])

            df_list.append(pd.DataFrame(to_write, index=[0]))

        df_list = pd.concat(df_list)
        with open('analysis_parameters', 'a') as f:
            df_list.to_csv(f, header=True, sep='\t', index=False)


        filename = 'results_per_mask.csv'
        os.rename('analysis_parameters', filename)

    # If there was only one mask, report scores per subject
    else:
        means = np.round(dfs.mean(axis=0), 3)
        dfs = dfs.append(means, ignore_index=True)

        with open('analysis_parameters', 'a') as f:
            dfs.to_csv(f, header=True, sep='\t', index=False)

        filename = 'results_per_sub.csv'
        os.rename('analysis_parameters', filename)

    if inputs['cluster_args']['do_clust']:
        [os.remove(p) for p in glob.glob('*Clustered*')]

    [os.remove(p) for p in glob.glob('*averaged*')]

    fs_files = glob.glob('*fs.nii.gz')
    vox_files = glob.glob('*vox.nii.gz')
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

    confmats = glob.glob('*HWW*npy')
    all_cms = np.zeros((len(confmats), 3, 3))
    for i, confmat in enumerate(confmats):
        all_cms[i, :, :] = np.load(confmat)

    np.save('all_confmats', all_cms)
    [os.remove(p) for p in glob.glob('*HWW*.npy')]
    os.system('cat %s' % filename)


if __name__ == '__main__':
    sys.path.append('/home/c6386806/LOCAL/Analysis_scripts')

    # Information about which data to use
    home = os.path.expanduser('~')
    feat_dir = opj(home, 'DecodingEmotions_tmp2')
    ROI_dir = opj(home, 'ROIs', 'Harvard_Oxford_atlas')
    os.chdir(feat_dir)
    identifier = 'merged'

    mvp_dir = opj(os.getcwd(), 'mvp_mats')
    header_dirs = sorted(glob.glob(opj(mvp_dir, '*%s*cPickle' % identifier)))
    data_dirs = sorted(glob.glob(opj(mvp_dir, '*%s*hdf5' % identifier)))
    subject_dirs = zip(header_dirs, data_dirs)

    # dictionary with possible masks (based on Harvard-Oxford cortical atlas)
    mask_file = dict([('bilateral', sorted(glob.glob(opj(ROI_dir, 'bilateral', '*nii.gz*')))),
                      ('unilateral', sorted(glob.glob(opj(ROI_dir, 'unilateral', '*nii.gz*')))),
                      ('wholebrain', glob.glob(opj(ROI_dir, 'GrayMatter.nii.gz')))])

    # Parameters for classification
    inputs = {# General paramters
              'clf': svm.SVC(kernel='linear'),
              'iterations': 2000,
              'n_test': 4,
              'mask_file': mask_file['wholebrain'],
              'cv_method': StratifiedShuffleSplit,
              'score_method': 'trial_based',
              'subject_timer': 'HWW_012',
              'save_feat_corrs': False,
              'n_cores': len(subject_dirs),

              # Feature selection parameters
              'fs_method': SelectAboveZvalue,
              'fs_arg': 1.5,
              'fs_doubledip': False,
              'fs_average': False,

              # Cluster parameters
              'cluster_args': {'do_clust': True,
                               'minimum': 40,
                               'cleanup': False,
                               'test_demean': False},


              # Other feature transformation parameters
              'demean_patterns': False,
              'average_patterns': False,

              # PCA stuff
              'pca_args': {'do_pca': False,
                           'n_comp': 1,
                           'start_idx': 1}}

    # TO IMPLEMENT: set overlapping voxels in ROI-average analysis to 0

    # RUN ANALYSIS
    Parallel(n_jobs=inputs['n_cores']) \
        (delayed(mvp_classify)(sub_dir, inputs) for sub_dir in subject_dirs)

    # Get individual results and average
    average_classification_results(inputs)
