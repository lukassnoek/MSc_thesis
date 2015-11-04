from __future__ import division
import glob
import os
import sys
import cPickle
import pandas as pd
import numpy as np
import nibabel as nib
import h5py
import progressbar as pbar
from os.path import join as opj
from sklearn.cross_validation import StratifiedShuffleSplit
from joblib import Parallel, delayed
from scipy.stats import f_oneway
from sklearn.base import TransformerMixin
from glm2mvpa import MVPHeader
import itertools as itls
from scipy.stats import t as t2p

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


def main(inputs):
    subject_dirs = inputs['subject_dirs']
    iterations = inputs['iterations']
    n_test = inputs['n_test']
    mask_file = inputs['mask_file']
    fs_method = inputs['fs_method']
    fs_arg = inputs['fs_arg']
    cv_method = inputs['cv_method']
    crossval = inputs['crossval']
    selector = fs_method(fs_arg)

    df = pd.DataFrame(columns=('subject', 'mask', 'F'))

    for sub_dir in subject_dirs:

        # Unpacking subject-data into header and actual data
        header_path, data_path = sub_dir
        mvp = cPickle.load(open(header_path))
        gm_data = h5py.File(data_path, 'r')['data'][:]

        for cMask, roi in enumerate(mask_file):

            mask_mask = nib.load(roi).get_data() > mvp.mask_threshold
            mask_mask = np.reshape(mask_mask, mvp.mask_index.shape)
            mask_overlap = mask_mask.astype(int) + mvp.mask_index.astype(int)

            # Update MVPHeader
            mvp.submask_index = (mask_overlap == 2)[mvp.mask_index]
            mvp.mask_name = os.path.basename(roi)[:-7]
            mvp.data = gm_data[:, mvp.submask_index]
            mvp.n_features = np.sum(mvp.submask_index)

            print "%s, processing mask: %s (%i out of %i)" % \
                  (mvp.subject_name, mvp.mask_name, cMask + 1, len(mask_file))

            if crossval:

                folds = cv_method(mvp.num_labels, iterations, n_test * mvp.n_class,
                                  random_state=0)

                Fstats = np.zeros(iterations)

                for i, (train_idx, test_idx) in enumerate(folds):
                    train_data = mvp.data[train_idx, :]
                    test_data = mvp.data[test_idx, :]
                    train_labels = mvp.num_labels[train_idx]
                    test_labels = mvp.num_labels[test_idx]

                    selector.fit(train_data, train_labels)
                    selector.transform(test_data)

                    test_data = np.mean(test_data, axis=1)
                    F = f_oneway(*[test_data[test_labels == gr] for gr in
                                  np.unique(test_labels)])

                    Fstats[i] = F[0]

                    df_tmp = pd.DataFrame({'subject': mvp.subject_name, 'mask': mvp.mask_name,
                                       'F': np.mean(Fstats)}, index=[0])
                    df = df.append(df_tmp)

            else:
                mvp.data = np.mean(mvp.data, axis=1)
                F = f_oneway(*[mvp.data[mvp.num_labels == gr] for gr in
                               np.unique(mvp.num_labels)])
                df_tmp = pd.DataFrame({'subject': mvp.subject_name, 'mask': mvp.mask_name,
                                       'F': F[0]}, index=[0])
                df = df.append(df_tmp)

    bygroup_mask = df.groupby(['mask'])
    agg_df = bygroup_mask['F'].agg([np.mean, np.std])
    agg_df['tval'] = (agg_df['mean'] - 1) / (agg_df['std'] / np.sqrt(12))
    agg_df['pval'] = [t2p.sf(tval, 11) for tval in agg_df['tval']]
    print agg_df

    with open('ROI_ttest_results.csv', 'w') as f:
        agg_df.to_csv(f, header=True, sep='\t', index=True)


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
    mask_file = dict([('bilateral', sorted(
        glob.glob(opj(ROI_dir, 'bilateral', '*nii.gz*')))),
                      ('unilateral', sorted(
                          glob.glob(opj(ROI_dir, 'unilateral', '*nii.gz*')))),
                      ('wholebrain',
                       glob.glob(opj(ROI_dir, 'GrayMatter.nii.gz')))])

    # Parameters for classification
    inputs = {  # General paramters
                'iterations': 1000,
                'n_test': 4,
                'mask_file': mask_file['unilateral'],
                'cv_method': StratifiedShuffleSplit,
                'subject_timer': 'HWW_012',
                'subject_dirs': subject_dirs,
                # Feature selection parameters
                'fs_method': SelectAboveZvalue,
                'fs_arg': 1,
                'crossval': True}

    # TO IMPLEMENT: set overlapping voxels in ROI-average analysis to 0
    # RUN ANALYSIS
    main(inputs)
