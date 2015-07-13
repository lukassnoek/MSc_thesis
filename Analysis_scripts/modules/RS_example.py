# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import os
import nibabel as nib
import sys
import numpy as np

sys.path.append('/home/lukas/Dropbox/ResMas_UvA/Thesis/Git/Analysis_scripts/modules')
import FSLutilities as fsl

os.chdir('/media/lukas/Data/RS_project/')

fDir = '/media/lukas/Data/RS_project/hww001.feat'
reg_dir = fDir + '/reg'
stats_dir = fDir + '/stats'
mni_file = reg_dir + '/standard.nii.gz'
premat_file = reg_dir + '/example_func2highres.mat'
warp_file = reg_dir + '/highres2standard_warp.nii.gz'
to_transform = 'hww001.feat/filtered_func_data.nii.gz'
out_name = 'filtered_func_data_MNI.nii.gz'           

os.system("applywarp --ref=" + mni_file + 
                      " --in=" + to_transform + 
                      " --out=" + out_name + 
                      " --warp=" + warp_file +
                      " --premat=" + premat_file +
                      " --interp=trilinear")

func = nib.load('filtered_func_data_MNI.nii.gz').get_data()
csf = nib.load('/usr/share/fsl/5.0/data/standard/MNI152_T1_2mm_VentricleMask.nii.gz').get_data()
csf_bool = csf > 0

func_csf = func[np.tile(csf_bool,(csf_bool.shape, 301))]