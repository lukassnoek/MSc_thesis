# -*- coding: utf-8 -*-
"""
Script to run main classification analysis.
Dynamic Affect project. 
"""

################################ SETUP dirs ################################
import sys
import os
import glob
from os.path import join as opj
import multiprocessing as mp

home = os.path.expanduser("~")
script_dir = os.path.join(home, 'LOCAL', 'Analysis_scripts')
sys.path.append(script_dir)    

from modules.main_classify import mvp_classify
feat_dir = os.path.join(home,'DecodingEmotions')
ROI_dir = opj(home,'ROIs')

os.chdir(feat_dir)

################################ SETUP params ################################
OFC_mask = opj(ROI_dir, 'OrbitofrontalCortex.nii.gz')
FL_mask = opj(ROI_dir, 'FrontalLobe.nii.gz')

# Parameters
identifier = 'merged'
iterations = 20
n_test = 2
zval = 2.3
method = 'fstat'
mask_file = OFC_mask

mvp_dir = opj(os.getcwd(), 'mvp_mats')
header_dirs = sorted(glob.glob(opj(mvp_dir, '*%s*cPickle' % identifier)))
data_dirs = sorted(glob.glob(opj(mvp_dir, '*%s*hdf5' % identifier)))
subject_dirs = zip(header_dirs, data_dirs)

################################ SETUP multiproc #############################

from modules.main_classify import create_results_log
create_results_log(iterations, zval, n_test, method, mask_file)

pool = mp.Pool(processes=len(subject_dirs))
results = [pool.apply_async(mvp_classify, args=(sub_dir, mask_file, iterations,
                            n_test, zval, method)) for sub_dir in subject_dirs]

#mvp_classify(identifier, mask_file, iterations, n_test, zval, method)
