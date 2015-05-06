# -*- coding: utf-8 -*-
"""
Created on Mon May  4 13:14:56 2015

@author: lukas
"""

import os
import cPickle
import sys
sys.path.append('/home/lukas/Dropbox/ResMas_UvA/Thesis/Git/Analysis_scripts/modules/')

import glm2mvpa as g2m
import main_classify as main

os.chdir('/media/lukas/Data/Matlab2Python/FirstLevel')
mask = '/media/lukas/Data/Matlab2Python/FSL_masks/ROIs/Bilateral_masks/CingulateGyrus_post.nii.gz'
subject_stem = 'hww'
mask_threshold = 30
norm_method = 'univariate'

g2m.create_subject_mats(mask, subject_stem, mask_threshold,norm_method)

os.chdir('/media/lukas/Data/Matlab2Python/FirstLevel/mvpa_mats')
x=main.mvpa_classify(100,4)