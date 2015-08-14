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
# Parameters
identifier = 'merged'
iterations = 5
n_test = 2
zval = 2.3
method = 'fstat'
mask_file = glob.glob(opj(ROI_dir, 'Harvard_Oxford_atlas', 'bilateral', '*CingulateGyrus*'))

mvp_dir = opj(os.getcwd(), 'mvp_mats')
header_dirs = sorted(glob.glob(opj(mvp_dir, '*%s*cPickle' % identifier)))
data_dirs = sorted(glob.glob(opj(mvp_dir, '*%s*hdf5' % identifier)))
subject_dirs = zip(header_dirs, data_dirs)

################################ SETUP multiproc #############################

#from modules.main_classify import create_results_log
#create_results_log(iterations, zval, n_test, method)

pool = mp.Pool(processes=len(subject_dirs))
results = [pool.apply_async(mvp_classify, args=(sub_dir, mask_file, iterations,
                            n_test, zval, method)) for sub_dir in subject_dirs]
results = [p.get() for p in results]
print results
#mvp_classify(identifier, mask_file, iterations, n_test, zval, method)
