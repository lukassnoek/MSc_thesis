# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 17:36:06 2015

@author: lsnoek1
"""

################################ SETUP dirs ################################
import sys
import os
<<<<<<< HEAD
home = os.path.expanduser("~")
script_dir = os.path.join(home,'LOCAL','Analysis_scripts')
sys.path.append(script_dir)    

################################ SETUP params ################################
from modules.glm2mvpa import create_subject_mats, merge_runs

ROI_dir = os.path.join(home,'ROIs')
=======

if 'Windows' in platform.platform():
    data_dir = os.path.join('Z:/','fMRI Projects','fMRI Project DynamicAffect')
    script_dir = os.path.join(home,'Dropbox','ResMas_UvA','Thesis','Git',
                              'Analysis_scripts')
    
if 'precise' in platform.platform():
    data_dir = os.path.join(home,'.gvfs','storage on bigbrother','fMRI Projects','fMRI Project DynamicAffect')
    script_dir = os.path.join(home,'Git','Analysis_scripts')

if 'vivid' in platform.platform():
    data_dir = os.path.join('/run/user/1000/gvfs',
                            'smb-share:server=bigbrother.fmg.uva.nl,share=fmri_projects$',
                            'fMRI Project DynamicAffect')
    script_dir = os.path.join(home,'Dropbox','ResMas_UvA','Thesis','Git',
                              'Analysis_scripts')    

sys.path.append(script_dir)    

################################ SETUP params ################################
from modules.glm2mvpa import create_subject_mats

ROI_dir = os.path.join(data_dir,'ROIs')
>>>>>>> 83d4de9352244426d9478187d00161c4bd8841dc
GM_mask = os.path.join(ROI_dir, 'GrayMatter.nii.gz')
OFC_mask = os.path.join(ROI_dir, 'OrbitofrontalCortex.nii.gz')
FL_mask = os.path.join(ROI_dir,'FrontalLobe.nii.gz')
MNI_mask = os.path.join(ROI_dir, 'MNI152_T1_2mm_brain.nii.gz')

feat_dir = os.path.join(home,'DecodingEmotions')
os.chdir(feat_dir)

mask = GM_mask
subject_stem = 'HWW'
mask_threshold = 30
remove_class = []
grouping = []
norm_method = 'univariate'

create_subject_mats(mask, subject_stem, mask_threshold, remove_class,
<<<<<<< HEAD
                    grouping=grouping, norm_method='univariate')
=======
                             grouping = grouping, norm_method = 'univariate')
>>>>>>> 83d4de9352244426d9478187d00161c4bd8841dc

merge_runs()