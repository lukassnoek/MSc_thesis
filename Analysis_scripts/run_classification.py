# -*- coding: utf-8 -*-
"""
Script to run main classification analysis on data from multiple participants,
using the multiprocessing package which allows to map the mvp_classify function
onto multiple functions.

Results, per mask averaged over subjects, are written to the file
'results_summary'.

Dynamic Affect project, 2015
Lukas Snoek
"""

################################ SETUP dirs ################################
import sys
import os
import glob
from os.path import join as opj
import multiprocessing as mp
import pandas as pd
import numpy as np

home = os.path.expanduser("~")
script_dir = os.path.join(home, 'LOCAL', 'Analysis_scripts')
sys.path.append(script_dir)    

from modules.main_classify import mvp_classify
feat_dir = opj(home, 'DynamicAffect_MV/FSL_FirstLevel_Posttest')
ROI_dir = opj(home, 'ROIs')

os.chdir(feat_dir)

################################ SETUP params ################################
# Parameters
identifier = ''
iterations = 100
n_test = 1
zval = 1.5
vs_method = 'fstat'
mask_file = sorted(glob.glob(opj(ROI_dir, 'Harvard_Oxford_atlas', 'bilateral', '*.nii.gz')))
# alleen GM

mvp_dir = opj(os.getcwd(), 'mvp_mats')
header_dirs = sorted(glob.glob(opj(mvp_dir, '*%s*cPickle' % identifier)))
data_dirs = sorted(glob.glob(opj(mvp_dir, '*%s*hdf5' % identifier)))
subject_dirs = zip(header_dirs, data_dirs)

################################ SETUP multiproc #############################

from modules.main_classify import create_results_log
create_results_log(iterations, zval, n_test, vs_method)

pool = mp.Pool(processes=len(subject_dirs))
results = [pool.apply_async(mvp_classify, args=(sub_dir, mask_file, iterations,
                            n_test, zval, vs_method)) for sub_dir in subject_dirs]
results = [p.get() for p in results]
results = pd.concat(results)

masks = np.unique(results['mask'])

total_df = []
for mask in masks:
    score = np.mean(results['score'][results['mask'] == mask]).round(3)
    df = {'mask': mask,
          'score': score}
    total_df.append(pd.DataFrame(df, index=[1]))

to_write = pd.concat(total_df)
with open('results_summary', 'a') as f:
    to_write.to_csv(f, header=False, sep='\t', index=False)