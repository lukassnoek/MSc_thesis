# -*- coding: utf-8 -*-
"""
Created on Mon May  4 13:14:56 2015

@author: lukas
"""

import os
import cPickle
import sys
sys.path.append('/home/lukas/Dropbox/ResMas_UvA/Thesis/Git/Analysis_scripts/modules/')

import FSLutilities
os.chdir('/media/lukas/Data/Matlab2Python/FirstLevel')

FSLutilities.call_flirt('cope')

import glm2mvpa as g2m
import main_classify as main

subject_stem = 'HWW'
mask_threshold = 30
norm_method = 'univariate'

mask = '/media/lukas/Data/Matlab2Python/FSL_masks/WholeBrain/GrayMatter.nii.gz'
mask2 = '/media/lukas/Data/Matlab2Python/FSL_masks/ROIs/Bilateral_masks/CingulateGyrus_post.nii.gz'
g2m.create_subject_mats(mask2, subject_stem, mask_threshold,norm_method)

for i in range(1000):
    train_idx = np.invert(draw_random_subsample(mvpa, 4))
    if np.sum(main.select_voxels(mvpa,train_idx, 2.3)) == 0:
        print "too little"

x,vc=main.mvpa_classify(100,4, 2.3)
