# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 17:36:06 2015

@author: lsnoek1
"""

################################ SETUP dirs ################################
import sys
import os

home = os.path.expanduser("~")
script_dir = os.path.join(home,'LOCAL','Analysis_scripts')
sys.path.append(script_dir)    
ROI_dir = os.path.join(home,'ROIs')

################################ SETUP params ################################
from modules.glm2mvpa import create_subject_mats, merge_runs

GM_mask = os.path.join(ROI_dir, 'GrayMatter.nii.gz')
OFC_mask = os.path.join(ROI_dir, 'OrbitofrontalCortex.nii.gz')
FL_mask = os.path.join(ROI_dir,'FrontalLobe.nii.gz')
MNI_mask = os.path.join(ROI_dir, 'MNI152_T1_2mm_brain.nii.gz')

feat_dir = os.path.join(home,'DecodingEmotions')
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