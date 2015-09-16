# -*- coding: utf-8 -*-
"""
Module to create subject-specific mvp matrices of trials X voxels.
Contains:
1. mvp_mat class;
2. create_subject_mats (main mvp_mat creation);
3. Some uninteresting but necessary file parsing/transforming functions

The creat_subject_mats function does the following for each subject:
1. Loads in mni-transformed first-level COPEs
2. Indexes vectorized copes with specified mask (e.g. ROI/gray matter)
3. Normalizes COPEs by their variance (sqrt(VARCOPE)); this will be extended 
with a multivariate normalization technique in the future
4. Initializes the result as an mvp_mat
5. Saves subject-specific mvp_mat as .cpickle file 

Lukas Snoek, master thesis Dynamic Affect, 2015
"""

__author__ = "Lukas Snoek"

import os
import numpy as np
import nibabel as nib
import glob
import csv
import cPickle
import fnmatch
import shutil
import h5py
import sys

from os.path import join as opj
from sklearn import preprocessing as preproc
from itertools import chain, izip
import nipype.interfaces.fsl as fsl

class MVPHeader:
    """
    Contains info about multivariate pattern fMRI data.
    """

    def __init__(self, data, subject_name, mask_name, mask_index, mask_shape,
                 mask_threshold, class_labels, num_labels, grouping):

        # Primary data
        self.data = None
        self.subject_name = subject_name        # subject name
        self.n_features = data.shape[1]
        self.n_trials = data.shape[0]

        # Information about mask
        self.mask_name = mask_name              # Name of nifti-file
        self.submask_name = None
        self.mask_index = mask_index            # index relative to MNI
        self.submask_index = None
        self.mask_shape = mask_shape            # shape of mask (usually mni)
        self.mask_threshold = mask_threshold

        # Information about condition/class
        self.class_labels = np.asarray(class_labels)        # class-labels trials
        self.class_names = np.unique(self.class_labels)
        self.n_class = len(np.unique(num_labels))
        self.n_inst = [np.sum(cls == num_labels) for cls in np.unique(num_labels)]

        self.class_idx = [num_labels == cls for cls in np.unique(num_labels)]
        self.trial_idx = [np.where(num_labels == cls)[0] for cls in np.unique(num_labels)]

        self.num_labels = num_labels
        self.grouping = grouping

    def normalize(self, style):
        pass


def extract_class_vector(sub_path, remove_class):
    """ Extracts class of each trial and returns a vector of class labels."""

    sub_name = os.path.basename(os.path.normpath(sub_path))
    to_parse = opj(sub_path, 'design.con')

    # Read in design.con
    if os.path.isfile(to_parse):
        with open(to_parse, 'r') as con_file:
            con_read = csv.reader(con_file, delimiter='\t')
            class_labels = []

            # Extract class of trials until /NumWaves
            for line in con_read:
                if line[0] == '/NumWaves':
                    break
                else:
                    class_labels.append(line[1])

        # Remove classes/trials based on remove_class list
        remove_idx = []
        for match in remove_class:
            to_remove = fnmatch.filter(class_labels, '*%s*' % match)

            for name in to_remove:
                remove_idx.append(class_labels.index(name))

        remove_idx = list(set(remove_idx))
        [class_labels.pop(idx) for idx in sorted(remove_idx, reverse=True)]

        class_labels = [s.split('_')[0] for s in class_labels]

        return class_labels, remove_idx

    else:
        print('There is no design.con file for ' + sub_name)


def create_subject_mats(sub_path, inputs):
    """ 
    Creates subject-specific mvp matrices, initializes them as an
    mvp_mat object and saves them as a cpickle file.
    
    Args: 
    firstlevel_dir  = directory with individual firstlevel data
    mask            = mask to index constrasts, either 'fstat' or a specific 
                      ROI. The exact name should be given 
                      (e.g. 'graymatter.nii.gz').
    subject_stem    = project-specific subject-prefix
    mask_threshold  = min. threshold of probabilistic FSL mask
    remove_class    = list with strings to match trials/classes to which need
                      to be removed (e.g. noise regressors)
   
    Returns:
    Nothing, but creates a dir ('mvp_mats') with individual pickle files.
    
    Lukas Snoek    
    """

    # Unpack
    mask = inputs['mask']
    mask_threshold = inputs['mask_threshold']
    remove_class = inputs['remove_class']
    grouping = inputs['grouping']
    norm_method = inputs['norm_method']
    transform_res4d = inputs['transform_res4d']
    scaler = inputs['scaler']

    # Load mask, create index
    mask_name = os.path.basename(mask)
    mask_vol = nib.load(mask)
    mask_shape = mask_vol.shape
    mask_index = mask_vol.get_data().ravel() > mask_threshold
    n_features = np.sum(mask_index)

    n_feat = len(glob.glob(opj(os.path.dirname(sub_path), '*.feat')))

    # Extract class vector (see definition below)
    class_labels, remove_idx = extract_class_vector(sub_path, remove_class)

    # Grouping
    if len(grouping) == 0:
        grouping = np.unique(class_labels)

    num_labels = np.zeros(len(class_labels))
    for i, group in enumerate(grouping):

        if type(group) == list:
            matches = []
            for g in group:
                matches.append(fnmatch.filter(class_labels, '*%s*' % g))
            matches = [x for y in matches for x in y]
        else:
            matches = fnmatch.filter(class_labels, '*%s*' % group)
            matches = list(set(matches))

        for match in matches:
            for k, lab in enumerate(class_labels):
                if match == lab:
                    num_labels[k] = i + 1

    sub_name = os.path.basename(sub_path)

    print 'Processing ' + os.path.basename(sub_name) + ' ... '

    if transform_res4d:
        os.chdir(sub_path)

        # Transform res4d
        res4d = opj(sub_path, 'stats', 'res4d.nii.gz')
        mat_file = opj(sub_path, 'reg', 'example_func2highres.mat')
        out_file = opj(mat_dir, '%s_res4d.nii.gz' % sub_name[:-5])
        ref_file = opj(sub_path, 'reg', 'standard.nii.gz')

        apply_xfm = fsl.ApplyXfm()
        apply_xfm.inputs.in_file = res4d
        apply_xfm.inputs.in_matrix_file = mat_file
        apply_xfm.inputs.out_file = out_file
        apply_xfm.inputs.reference = ref_file
        apply_xfm.inputs.apply_xfm = True
        apply_xfm.interp = 'nearestneighbor'
        _ = apply_xfm.run()
        os.chdir('..')

    # Generate and sort paths to stat files (COPEs/tstats)
    stat_dir = opj(sub_path, 'reg_standard')
    if not os.path.isdir(stat_dir):
        os.mkdir(stat_dir)
        stat_dir = opj(sub_path, 'stats')

    if norm_method == 'nothing':
        stat_paths = glob.glob(opj(stat_dir, 'tstat*.nii.gz'))
    else:
        stat_paths = glob.glob(opj(stat_dir, 'cope*.nii.gz'))

    stat_paths = sort_stat_list(stat_paths) # see function below
    [stat_paths.pop(idx) for idx in sorted(remove_idx, reverse=True)]
    n_stat = len(stat_paths)

    varcopes = glob.glob(opj(stat_dir,'varcope*.nii.gz'))
    varcopes = sort_stat_list(varcopes)
    [varcopes.pop(idx) for idx in sorted(remove_idx, reverse=True)]

    if not n_stat == len(class_labels):
        msg = 'The number of trials do not match the number of class labels'
        raise ValueError(msg)

    if n_stat == 0:
        msg = 'There are no valid COPES/tstats for subject %s.' % sub_name
        raise ValueError(msg)

    # Transform to mni space if necessary
    if os.path.basename(stat_dir) == 'stats':
        os.chdir(sub_path)
        print "Transforming COPES to MNI for %s." % sub_name
        ref_file = opj(sub_path, 'reg', 'standard.nii.gz')
        field_file = opj(sub_path, 'reg', 'example_func2standard_warp.nii.gz')
        out_dir = opj(sub_path, 'reg_standard')

        to_transform = zip(stat_paths, varcopes)

        for stat, varc in to_transform:

            out_file = opj(out_dir, os.path.basename(stat))
            apply_warp = fsl.ApplyWarp()
            apply_warp.inputs.in_file = stat
            apply_warp.inputs.ref_file = ref_file
            apply_warp.inputs.field_file = field_file
            apply_warp.interp = 'trilinear'
            apply_warp.inputs.out_file = out_file
            apply_warp.run()

            out_file = opj(out_dir, os.path.basename(varc))
            apply_warp = fsl.ApplyWarp()
            apply_warp.inputs.in_file = varc
            apply_warp.inputs.ref_file = ref_file
            apply_warp.inputs.field_file = field_file
            apply_warp.interp = 'trilinear'
            apply_warp.inputs.out_file = out_file
            apply_warp.run()

        stat_dir = opj(sub_path, 'reg_standard')
        stat_paths = glob.glob(opj(stat_dir, 'cope*'))
        stat_paths = sort_stat_list(stat_paths) # see function below

        varcopes = glob.glob(opj(stat_dir, 'varcope*'))
        varcopes = sort_stat_list(varcopes) # see function below

        os.chdir('..')

    # Pre-allocate
    mvp_data = np.zeros([n_stat, n_features])

    # Load in data (COPEs)
    for i, path in enumerate(stat_paths):
        cope = nib.load(path).get_data()
        mvp_data[i,:] = np.ravel(cope)[mask_index]

    ''' NORMALIZATION OF VOXEL PATTERNS '''
    if norm_method == 'univariate':

        for i_trial, varcope in enumerate(varcopes):
            var = nib.load(varcope).get_data()
            var_sq = np.sqrt(var.ravel()[mask_index])
            mvp_data[i_trial,] = np.divide(mvp_data[i_trial,], var_sq)

    mvp_data[np.isnan(mvp_data)] = 0

    if scaler == 'minmax':
        sclr = preproc.MinMaxScaler(feature_range=(0, 1))
        mvp_data = sclr.fit_transform(mvp_data)
    elif scaler == 'ztrans':
        mvp_data = preproc.scale(mvp_data)
    else:
        raise ValueError('Did not pick an appropriate scaler.')

    # Initializing mvp_mat object, which will be saved as a pickle file
    to_save = MVPHeader(mvp_data, sub_name, mask_name, mask_index,
                        mask_shape, mask_threshold, class_labels,
                        num_labels, grouping)

    fn_header = opj(mat_dir, '%s_header.cPickle' % sub_name)
    fn_data = opj(mat_dir, '%s_data.hdf5' % sub_name)

    with open(fn_header, 'wb') as handle:
        cPickle.dump(to_save, handle)

    h5f = h5py.File(fn_data, 'w')
    h5f.create_dataset('data', data=mvp_data)
    h5f.close()

    print 'Done processing %s.' % sub_name


def sort_stat_list(stat_list):
    """
    Sorts list with paths to statistic files (e.g. COPEs, VARCOPES),
    which are often sorted wrong (due to single and double digits).
    This function extracts the numbers from the stat files and sorts 
    the original list accordingly.
    """
    num_list = []
    for path in stat_list:
        num = [str(s) for s in str(os.path.basename(path)) if s.isdigit()]
        num_list.append(int(''.join(num)))

    sorted_list = [x for y, x in sorted(zip(num_list, stat_list))]
    return(sorted_list)


def merge_runs():
    '''
    Merges mvp_mat objects from multiple runs. 
    Incomplete; assumes only two runs for now.
    '''

    header_paths = sorted(glob.glob(opj(os.getcwd(),'mvp_mats','*cPickle*')))
    header_paths = zip(header_paths[::2], header_paths[1::2])

    data_paths = sorted(glob.glob(opj(os.getcwd(),'mvp_mats','*hdf5*')))
    data_paths = zip(data_paths[::2], data_paths[1::2])

    sub_paths = zip(header_paths, data_paths)
    n_sub = len(sub_paths)

    for header, data in sub_paths:
        run1_h = cPickle.load(open(header[0]))
        run2_h = cPickle.load(open(header[1]))

        run1_d = h5py.File(data[0],'r')
        run2_d = h5py.File(data[1],'r')

        merged_grouping = run1_h.grouping
        merged_mask_index = run1_h.mask_index
        merged_mask_shape = run1_h.mask_shape
        merged_mask_name = run1_h.mask_name
        merged_mask_threshold = run1_h.mask_threshold
        merged_name = run1_h.subject_name.split('-')[0]

        merged_data = np.empty((run1_d['data'].shape[0] +
                                run2_d['data'].shape[0],
                                run1_d['data'].shape[1]))

        merged_data[::2,:] = run1_d['data'][:]
        merged_data[1::2,:] = run2_d['data'][:]

        merged_class_labels = list(chain.from_iterable(izip(run1_h.class_labels,
                                                            run2_h.class_labels)))

        merged_num_labels = list(chain.from_iterable(izip(run1_h.num_labels,
                                                          run2_h.num_labels)))

        to_save = MVPHeader(merged_data, merged_name, merged_mask_name,
                            merged_mask_index, merged_mask_shape,
                            merged_mask_threshold, merged_class_labels,
                            np.asarray(merged_num_labels), merged_grouping)

        fn = opj(os.getcwd(), 'mvp_mats', merged_name + '_header_merged.cPickle')
        with open(fn, 'wb') as handle:
            cPickle.dump(to_save, handle)

        fn = opj(os.getcwd(), 'mvp_mats', merged_name + '_data_merged.hdf5')
        h5f = h5py.File(fn, 'w')
        h5f.create_dataset('data', data=merged_data)
        h5f.close()

        print "Merged subject %s " % merged_name

    #os.system('rm %s' % opj(os.getcwd(), 'mvp_mats', '*WIPPM*.cPickle'))
    #os.system('rm %s' % opj(os.getcwd(), 'mvp_mats', '*WIPPM*.hdf5'))

if __name__ == '__main__':

    from joblib import Parallel, delayed

    home = os.path.expanduser('~')
    script_dir = opj(home, 'LOCAL', 'Analysis_scripts')
    sys.path.append(script_dir)
    ROI_dir = opj(home, 'ROIs')

    feat_dir = opj(home, 'DecodingEmotions')
    os.chdir(feat_dir)

    subject_stem = 'HWW'
    data_dir = opj(os.getcwd(), '*%s*' % subject_stem, '*.feat')
    subject_dirs = sorted(glob.glob(data_dir))
    mat_dir = opj(os.getcwd(), 'mvp_mats')

    if os.path.exists(mat_dir):
        shutil.rmtree(mat_dir)

    os.makedirs(mat_dir)

    # Parameters
    inputs = {}
    inputs['mask'] = opj(ROI_dir, 'GrayMatter.nii.gz')
    inputs['mask_threshold'] = 0
    inputs['remove_class'] = []
    inputs['grouping'] = []
    inputs['norm_method'] = 'univariate'
    inputs['transform_res4d'] = False
    inputs['scaler'] = 'ztrans' # or: 'ztrans'

    Parallel(n_jobs=len(subject_dirs)) \
        (delayed(create_subject_mats)(sub_dir, inputs) for sub_dir in subject_dirs)

    merge_runs()