# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 17:36:06 2015

@author: lsnoek1
"""

################################ SETUP dirs ################################
import sys
import platform
from os.path import expanduser
home = expanduser("~")
import os

if 'Windows' in platform.platform():
    data_dir = os.path.join('Z:/','fMRI Projects','fMRI Project DynamicAffect')
    script_dir = os.path.join(home,'Dropbox','ResMas_UvA','Thesis','Git',
                              'Analysis_scripts')
    
if 'precise' in platform.platform():
    data_dir = os.path.join(home,'.gvfs','storage on bigbrother')
    script_dir = os.path.join(home,'Git','Analysis_scripts')

if 'vivid' in platform.platform():
    data_dir = os.path.join('/run/user/1000/gvfs',
                            'smb-share:server=bigbrother.fmg.uva.nl,share=fmri_projects$',
                            'fMRI Project DynamicAffect')
    script_dir = os.path.join(home,'Dropbox','ResMas_UvA','Thesis','Git',
                              'Analysis_scripts')    

sys.path.append(script_dir)    

################################ SETUP params ################################
from Modules.glm2mvpa import create_subject_mats

ROI_dir = os.path.join(home,'Dropbox/ResMas_UvA/Thesis/ROIs')
GM_mask = os.path.join(ROI_dir, 'GrayMatter.nii.gz')
OFC_mask = os.path.join(ROI_dir, 'OrbitofrontalCortex.nii.gz')
FL_mask = os.path.join(ROI_dir,'FrontalLobe.nii.gz')
MNI_mask = os.path.join(ROI_dir, 'MNI152_T1_2mm_brain.nii.gz')

feat_dir = os.path.join(data_dir,'DecodingEmotions')
os.chdir(feat_dir)

mask = GM_mask
subject_stem = 'HWW'
mask_threshold = 10
remove_class = []
grouping = []
norm_method = 'univariate'

glm2mvpa.create_subject_mats(mask, subject_stem, mask_threshold, remove_class,
                             grouping = grouping, norm_method = 'univariate')

#glm2mvpa.merge_runs()