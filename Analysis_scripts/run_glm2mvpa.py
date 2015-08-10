# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 17:36:06 2015

@author: lsnoek1
"""

# Set-up
from os.path import expanduser
home = expanduser("~")
import os

import sys
script_dir = os.path.join(home,'Dropbox','ResMas_UvA','Thesis','Git','Analysis_scripts')
sys.path.append(script_dir)

if sys.platform[0:5] == 'linux':
    data_dir = os.path.join('/run/user/1000/gvfs',
    'smb-share:server=bigbrother.fmg.uva.nl,share=fmri_projects$',
    'fMRI Project DynamicAffect')
else:
    data_dir = os.path.join('Z:/','fMRI Projects','fMRI Project DynamicAffect')

os.chdir(data_dir)

# Paramters
ROI_dir = os.path.join(home,'Dropbox/ResMas_UvA/Thesis/ROIs')
GM_mask = os.path.join(ROI_dir, 'GrayMatter.nii.gz')
OFC_mask = os.path.join(ROI_dir, 'OrbitofrontalCortex.nii.gz')
FL_mask = os.path.join(ROI_dir,'FrontalLobe.nii.gz')
MNI_mask = os.path.join(ROI_dir, 'MNI152_T1_2mm_brain.nii.gz')

feat_dir = os.path.join(data_dir,'Firstlevel_posttest')
os.chdir(feat_dir)

from Modules import glm2mvpa

mask = GM_mask
subject_stem = 'da'
mask_threshold = 10
remove_class = ['eval','loc']
grouping = [['pos','neg'],'neu']
norm_method = 'univariate'

glm2mvpa.create_subject_mats(mask, subject_stem, mask_threshold, remove_class,
                             grouping = grouping, norm_method = 'univariate')

