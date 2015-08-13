# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 17:36:06 2015

@author: lsnoek1
"""

################################ SETUP dirs ################################
import sys
import os
from os.path import join as opj

home = os.path.expanduser("~")
script_dir = opj(home,'LOCAL','Analysis_scripts')
sys.path.append(script_dir)    
ROI_dir = opj(home,'ROIs')

################################ SETUP params ################################
from modules.glm2mvpa import create_subject_mats, merge_runs

GM_mask = opj(ROI_dir, 'GrayMatter.nii.gz')
OFC_mask = opj(ROI_dir, 'OrbitofrontalCortex.nii.gz')
FL_mask = opj(ROI_dir, 'FrontalLobe.nii.gz')
MNI_mask = opj(ROI_dir, 'MNI152_T1_2mm_brain.nii.gz')

feat_dir = opj(home, 'DecodingEmotions')
os.chdir(feat_dir)

mask = GM_mask
subject_stem = 'HWW'
mask_threshold = 10
remove_class = []
grouping = []
norm_method = 'univariate'

create_subject_mats(mask, subject_stem, mask_threshold, remove_class,
                    grouping=grouping, norm_method='univariate')
merge_runs()